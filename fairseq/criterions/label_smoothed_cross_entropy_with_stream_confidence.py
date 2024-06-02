# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn.functional as F
from fairseq import utils
@dataclass
class LabelSmoothedCrossEntropySimilarCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    confidence_weight: float = field(
        default=0.1,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True,stream_similar = None,stream_weight=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if stream_similar is not None:
        stream_mask = stream_similar.eq(0)
        nll_loss.masked_fill_(stream_mask,0.0)
        smooth_loss.masked_fill_(stream_mask,0.0)
    if stream_weight is not None:
        # print("nll loss shape:{}\tsmooth loss:{}".format(nll_loss.shape,smooth_loss.shape))
        # print("stream weight:{}".format(stream_weight.shape))
        nll_loss = nll_loss*stream_weight
        smooth_loss = smooth_loss*stream_weight
        # print("nll loss shape after:{}\tsmooth loss after:{}".format(nll_loss.shape,smooth_loss.shape))
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_stream_confidence", dataclass=LabelSmoothedCrossEntropySimilarCriterionConfig
)
class LabelSmoothedCrossEntropyStreamConfidenceCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        confidence_weight = 0.1,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.confident_weight = confidence_weight
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss,golden_trans_loss,stream_trans_loss,confidence_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # loss,enc_correct_num,dec_correct_num,enc_all_num,dec_all_num,stream_trans_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "confidence_loss":confidence_loss.data,
            "golden_trans_loss":golden_trans_loss.data,
            "stream_trans_loss":stream_trans_loss.data,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):


        pred_dec_similar = net_output[1]['pred_confidence'].squeeze(-1)
        # pred_dec_similar = net_output[1]['pred_dec_similar'].squeeze(-1)
        stream_decoder_out = net_output[1]['stream_decoder_out']


        # loss = enc_similar_loss+dec_similar_loss
        # return loss, nll_loss
        golden_lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        golden_trans_loss, nll_loss = label_smoothed_nll_loss(
            golden_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        stream_out=F.softmax(stream_decoder_out, dim=-1)
        golden_out = F.softmax(net_output[0],dim=-1)
        c = pred_dec_similar.unsqueeze(-1)
        confi_stream_out = c*stream_out + (1.0-c)*golden_out
        confi_stream_out = torch.log(confi_stream_out)
        confi_stream_out = confi_stream_out.view(-1, confi_stream_out.size(-1))
        stream_trans_loss, stream_nll_loss = label_smoothed_nll_loss(
            confi_stream_out,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        #    stream_similar = pred_dec_label.view(-1,1),
        #   stream_weight=stream_weight
        )
        confidence_loss = torch.sum(-torch.log(pred_dec_similar))
        loss = golden_trans_loss + stream_trans_loss + self.confident_weight*confidence_loss
        
        return loss,golden_trans_loss,stream_trans_loss,confidence_loss


    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        golden_trans_loss_sum = sum(log.get("golden_trans_loss", 0) for log in logging_outputs)
        stream_trans_loss_sum = sum(log.get("stream_trans_loss", 0) for log in logging_outputs)
        confidence_loss_sum = sum(log.get("confidence_loss", 0) for log in logging_outputs)
        
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "golden_trans_loss", golden_trans_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "stream_trans_loss", stream_trans_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar(
            "confidence_loss", confidence_loss_sum / sample_size / math.log(2), sample_size, round=3
        )          
        # metrics.log_derived(
        #     "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        # )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
