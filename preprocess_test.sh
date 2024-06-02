src=$1
tgt=$2
num=$3
knum=$4
langpair=$src-$tgt-wmt22
# DATA_PATH=../../../dataset/de-en
# DATA_PATH=../../../dataset/en-de/test
# SIMUL_DATA_PATH=../../../dataset/
DATA_PATH=../../../dataset/$langpair
DST_PATH=$DATA_PATH/preprocess-$num
DICT_PATH=$DATA_PATH
python preprocess.py --source-lang $src \
                     --target-lang $tgt \
                     --testpref $DATA_PATH/$num.bpe \
                     --destdir  $DST_PATH \
                     --srcdict  $DICT_PATH/dict.zh.45000 \
                     --tgtdict  $DICT_PATH/dict.en.35000
# vocab.zh-en-for-dinu.32k