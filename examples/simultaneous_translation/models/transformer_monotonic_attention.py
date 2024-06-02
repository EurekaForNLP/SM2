# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, NamedTuple, Optional
import random

import torch
import torch.nn as nn
from examples.simultaneous_translation.modules.monotonic_transformer_layer import (
    TransformerMonotonicDecoderLayer,
    TransformerMonotonicEncoderLayer,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderBase, 
    TransformerDecoderBase,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
    tiny_architecture
)
from torch import Tensor

# DEFAULT_MAX_SOURCE_POSITIONS = 1024
# DEFAULT_MAX_TARGET_POSITIONS = 1024
READ_ACTION = 0
WRITE_ACTION = 1

TransformerMonotonicDecoderOut = NamedTuple(
    "TransformerMonotonicDecoderOut",
    [
        ("action", int),
        ("p_choose", Optional[Tensor]),
        ("attn_list", Optional[List[Optional[Dict[str, Tensor]]]]),
        ("encoder_out", Optional[Dict[str, List[Tensor]]]),
        ("encoder_padding_mask", Optional[Tensor]),
        ("inner_states",Optional[List[Optional[Tensor]]])
    ],
)


@register_model("transformer_unidirectional")
class TransformerUnidirectionalModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerMonotonicEncoder(args, src_dict, embed_tokens)
        ## 如果encoder存在预先训练好的需要加载的参数的话，加载参数
        model_path = args.pretrained_encoder_model_path
        if model_path is not None:
            state = torch.load(model_path)
            model_parm = state['model']
            new_encoder_parm = {}
            for k,v in model_parm.items():
                if 'encoder.' in k and 'teacher_encoder.' not in k:
                    # print("k:{}".format(k))
                    k = k.split('.')[1:]
                    k = '.'.join(k)
                    new_encoder_parm[k]=v
            encoder.load_state_dict(new_encoder_parm,strict=True)
            #冻结加载进来的模块的参数
            for para in encoder.parameters():
                para.requires_grad = False
        
        return encoder
        # return TransformerMonotonicEncoder(args, src_dict, embed_tokens)


@register_model("transformer_monotonic")
class TransformerModelSimulTrans(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

    @staticmethod
    def add_args(parser):
        super(TransformerModel,TransformerModel).add_args(parser)
        parser.add_argument(
            "--multipath", action="store_true", default=False, help="Multipath training"
        )

class TransformerMonotonicEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerMonotonicEncoderLayer(args)
                for i in range(args.encoder_layers)
            ]
        )


class TransformerMonotonicDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerMonotonicDecoderLayer(args)
                for _ in range(args.decoder_layers)
            ]
        )
        self.policy_criterion = getattr(args, "policy_criterion", "any")
        self.multipath = getattr(args,'multipath',False)
    def pre_attention(
        self,
        prev_output_tokens,
        encoder_out_dict: Dict[str, List[Tensor]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = encoder_out_dict["encoder_out"][0]

        if "encoder_padding_mask" in encoder_out_dict:
            encoder_padding_mask = (
                encoder_out_dict["encoder_padding_mask"][0]
                if encoder_out_dict["encoder_padding_mask"]
                and len(encoder_out_dict["encoder_padding_mask"]) > 0
                else None
            )
        else:
            encoder_padding_mask = None

        return x, encoder_out, encoder_padding_mask

    def post_attention(self, x):
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

    def clean_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
    ):
        """
        Clean cache in the monotonic layers.
        The cache is generated because of a forward pass of decoder has run but no prediction,
        so that the self attention key value in decoder is written in the incremental state.
        end_id is the last idx of the layers
        """
        if end_id is None:
            end_id = len(self.layers)

        for index, layer in enumerate(self.layers):
            if index < end_id:
                layer.prune_incremental_state(incremental_state)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,  # unused
        alignment_layer: Optional[int] = None,  # unused
        alignment_heads: Optional[int] = None,  # unsed
        train_threshold=None,
        step=None,
        waitk_test_lagging=None,
        return_all_cross_attn=False,
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # incremental_state = None
        assert encoder_out is not None
        (x, encoder_outs, encoder_padding_mask) = self.pre_attention(
            prev_output_tokens, encoder_out, incremental_state
        )
        attn = None
        inner_states = [x]
        attn_list: List[Optional[Dict[str, Tensor]]] = []
        p_choose = torch.tensor([1.0])


        if self.multipath and incremental_state is None and step is None:
            multipath_k = random.choice([1,3,5,7,9])
        else:
            multipath_k = None
        for i, layer in enumerate(self.layers):

            x, attn, _ = layer(
                x=x,
                encoder_out=encoder_outs,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None
                else None,
                train_threshold=train_threshold,
                step=step,
                multipath_k = multipath_k,
                waitk_test_lagging=waitk_test_lagging,
            )

            inner_states.append(x)
            attn_list.append(attn)

            if incremental_state is not None:
                if_online = incremental_state["online"]["only"]
                assert if_online is not None
                if if_online.to(torch.bool):
                    # Online indicates that the encoder states are still changing
                    assert attn is not None
                    if self.policy_criterion == "any":
                        # Any head decide to read than read
                        head_read = layer.encoder_attn._get_monotonic_buffer(incremental_state)["head_read"]
                        assert head_read is not None
                        if head_read.any():
                            # We need to prune the last self_attn saved_state
                            # if model decide not to read
                            # otherwise there will be duplicated saved_state
                            self.clean_cache(incremental_state, i + 1)

                            return x, TransformerMonotonicDecoderOut(
                                action=0,
                                p_choose=p_choose,
                                attn_list=None,
                                encoder_out=None,
                                encoder_padding_mask=None,
                            )

        x = self.post_attention(x)
        # if attn_list[-1]["beta"].shape[2] > 20:
        #     print(attn_list[-1]["beta"].shape)
        #     torch.save(attn_list[-1]["beta"],'beta.pt')
        return x, TransformerMonotonicDecoderOut(
            action=1,
            p_choose=p_choose,
            attn_list=attn_list,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            inner_states=inner_states
        )


@register_model_architecture("transformer_monotonic", "transformer_monotonic")
def base_monotonic_architecture(args):
    base_architecture(args)
    args.encoder_unidirectional = getattr(args, "encoder_unidirectional", False)

@register_model_architecture("transformer_unidirectional", "transformer_unidirectional")
def base_monotonic_architecture(args):
    base_architecture(args)


@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_iwslt_de_en"
)
def transformer_monotonic_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
    base_monotonic_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_vaswani_wmt_en_de_big"
)
def transformer_monotonic_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_vaswani_wmt_en_fr_big"
)
def transformer_monotonic_vaswani_wmt_en_fr_big(args):
    transformer_monotonic_vaswani_wmt_en_fr_big(args)


@register_model_architecture(
    "transformer_unidirectional", "transformer_unidirectional_iwslt_de_en"
)
def transformer_unidirectional_iwslt_de_en(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("transformer_monotonic", "transformer_monotonic_tiny")
def monotonic_tiny_architecture(args):
    tiny_architecture(args)
    base_monotonic_architecture(args)

@register_model("transformer_monotonic_with_knowledge_distillation")
class TransformerModelSimulTransWithKD(TransformerModel):
    def __init__(self,cfg, encoder,decoder,teacher_encoder,teacher_decoder,args,task):
        super().__init__(cfg,encoder,decoder)
        self.teacher_encoder = teacher_encoder
        self.teacher_decoder = teacher_decoder
        self.args = args
        self.task = task
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerConfig.from_namespace(args)
        
        """Build a new model instance."""

        # hacky fixes for issue with II
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(','))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(','))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        #如果存在知识蒸馏的teacher 的模型路径，则将teacher 的encoder和decoder
        #分别加载进来，同时将teacher 的参数冻结
        if cfg.knowledge_distillation is not None:
            
            # teacher_encoder_embed_tokens = cls.build_embedding(
            #     cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            # )
            # if cfg.share_all_embeddings:
            #     teacher_decoder_embed_tokens = teacher_encoder_embed_tokens
            # else:
            #     teacher_decoder_embed_tokens = cls.build_embedding(
            #     cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            # )
            teacher_encoder_embed_tokens = encoder_embed_tokens
            teacher_decoder_embed_tokens = decoder_embed_tokens
            teacher_encoder = cls.build_teacher_encoder(cfg, src_dict, teacher_encoder_embed_tokens)
            teacher_decoder = cls.build_teacher_decoder(cfg, tgt_dict, teacher_decoder_embed_tokens)

            # for para in encoder.parameters():
            #     para.requires_grad = False
            # for para in decoder.parameters():
            #     para.requires_grad = False
            # teacher_encoder=cls.load_model(cfg.knowledge_distillation,teacher_encoder,"encoder")

            # teacher_decoder=cls.load_model(cfg.knowledge_distillation,teacher_decoder,"decoder")
        else:
            teacher_decoder = None
            teacher_encoder = None
        if not cfg.share_all_embeddings:
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)

        return cls(cfg, encoder, decoder,teacher_encoder,teacher_decoder,args,task)
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

    @classmethod
    def build_teacher_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_teacher_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
    @classmethod
    def load_model(cls,model_path,load_module,prefix):
        #从给定路径加载预训练的模型的参数
        model_parm = torch.load(model_path)["model"]
        module_parm_dict=load_module.state_dict()
        new_model_parm = {}
        # print("module_parm_dict:{}".format(module_parm_dict.keys()))
        # print("model_parm_dict:{}".format(model_parm.keys()))
        for k,v in model_parm.items():
            if prefix in k:
                k = k.split('.')[1:]
                k ='.'.join(k)
                if k in module_parm_dict.keys():
                    new_model_parm[k]=v
            elif 'encoder' not in k and 'decoder' not in k:
                print(k)
        # for k in module_parm_dict.keys():
        #     if k not in new_model_parm.keys():
        #         print("\n\n\nthere are parameters don't get load!!!\n\n\n")
        #         print("it's:{}\n\n".format(k))
        # model_parm = {k: v for k, v in model_parm.items() if k in module_parm_dict}
        # print("model_pram_dict:{}".format(new_model_parm.keys()))
        # print("module_parm_dict:{}".format(module_parm_dict.keys()))
        # module_parm_dict.update(new_model_parm)
        load_module.load_state_dict(new_model_parm,strict=True)
        #冻结加载进来的模块的参数
        for para in load_module.parameters():
            para.requires_grad = False
            # para.requires_grad_(False)

        return load_module
    @staticmethod
    def add_args(parser):
        super(TransformerModel,TransformerModel).add_args(parser)
        parser.add_argument(
            "--multipath", action="store_true", default=False, help="Multipath training"
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_cross_attn: bool = True,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            return_all_cross_attn = return_all_cross_attn,
        )
        if self.cfg.knowledge_distillation is not None:
            teacher_encoder_out = self.teacher_encoder(
                    src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
                )
            teacher_decoder_out = self.teacher_decoder(
                    prev_output_tokens,
                    encoder_out=teacher_encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                    return_all_cross_attn=return_all_cross_attn,
                )
            return decoder_out,teacher_encoder_out,teacher_decoder_out
        else:
            return decoder_out
@register_model_architecture("transformer_monotonic_with_knowledge_distillation", "transformer_monotonic_with_knowledge_distillation")
def base_monotonic_architecture(args):
    base_architecture(args)
    args.encoder_unidirectional = getattr(args, "encoder_unidirectional", False)