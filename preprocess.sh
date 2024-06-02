DATA_PATH=path/to/unprocessed/data
DST_PATH=path/to/save/processed/data
python preprocess.py -s $src -t $tgt \
                     --trainpref $DATA_PATH/train.bpe \
                     --validpref $DATA_PATH/valid.bpe \
                     --testpref $DATA_PATH/test.bpe \
                     --destdir  $DST_PATH \
                     --workers 30