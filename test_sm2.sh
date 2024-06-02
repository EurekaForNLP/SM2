export CUDA_VISIBLE_DEVICES=$1
epoch=$2
confidence_threshold=$3 # the confidence threshold of confidence-based policy, larger threshold means longer latency and higher translation quality.
TEST_SET_PATH=/path/to/preprocessed_test_dataset #this data path should be the same as DST_PATH in preprocess.sh

# checkpoint averaging for model
src_lang=de
tgt_lang=en
langpair=$src_lang-$tgt_lang
setting=SM2-unid
modelfile=../model/$langpair-$setting
last_model_file=$modelfile/checkpoint$epoch.pt
python average_checkpoints.py --inputs ${modelfile} --num-epoch-checkpoints 5 \
    --output ${modelfile}/average-model.pt --last_file ${last_model_file}
file=${modelfile}/average-model.pt

#the decoding output dir
decoding_path=output/$src_lang-$tgt_lang/$setting-$confidence_threshold-epoch$epoch-test
mkdir $decoding_path

# generate translation
python sim_generate.py ${DST_PATH} --path ${file} \
    --confidence-decoding  --sacrebleu --scoring sacrebleu --remove-bpe \
    --confidence-threshold $confidence_threshold  \
    --batch-size 1 --beam 1  
