export CUDA_VISIBLE_DEVICES=$1
export model_dir=../model
src_lang=$2
tgt_lang=$3

SETTING=SM2-unid
confidence_weight=0.1
batch_max_tokens=8192
max_epochs=20
langpair=$src_lang-$tgt_lang
model_path=$model_dir/$langpair-$SETTING

# The '--arch' should be 'transformer_with_sm2_unidirectional' for SM2 with unidirectional encoder settings.
# If the used device supports bf16, '--bf16' is suggested.
# If source and target language share embeddings, use '--share-all-embeddings'
python train.py  ../dataset/$langpair/preprocess --max-epoch $max_epochs --source-lang $src_lang --target-lang $tgt_lang \
           --arch transformer_with_sm2_unidirectional \
           --share-all-embeddings \
           --confidence-weight $confidence_weight \
           --min-prefix-src-len 1 \
           --max-tokens $batch_max_tokens   \
           --optimizer adam  \
           --bf16 \
           --save-interval 1 \
           --lr 5e-4 --lr-scheduler inverse_sqrt \
            --warmup-init-lr 1e-07 --warmup-updates 4000 \
            --save-dir $model_path \
            --stop-min-lr 1e-09 \
            --criterion  label_smoothed_cross_entropy_stream_confidence  \
            --label-smoothing 0.1