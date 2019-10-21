#!/usr/bin/env bash

SOURCE_DIR=../Data/ATIS

declare -a epochs=(80 45 30 15)

declare -a checkpoint=(checkpoint_atis_80_geo_50_transfer
                       checkpoint_atis_45_geo_50_transfer
                       checkpoint_atis_30_geo_50_transfer
                       checkpoint_atis_15_geo_50_transfer)

declare -a output=(Output/seq2seq_attention_output_atis_80_geo_50_transfer.txt
                   Output/seq2seq_attention_output_atis_45_geo_50_transfer.txt
                   Output/seq2seq_attention_output_atis_30_geo_50_transfer.txt
                   Output/seq2seq_attention_output_atis_15_geo_50_transfer.txt)

declare -a source=(Output/seq2seq_attention_output_atis_80_transfer.txt
                   Output/seq2seq_attention_output_atis_45_transfer.txt
                   Output/seq2seq_attention_output_atis_30_transfer.txt
                   Output/seq2seq_attention_output_atis_15_transfer.txt)

declare -a model=(../Model/checkpoint_atis_80_geo_50_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_45_geo_50_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_30_geo_50_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_15_geo_50_transfer/model_seq2seq_attention)

#cd ../Model

export PYTHONPATH=${PYTHONPATH}:../

TARGET_DIR=../Data/GEO_50

#for i in ${!epochs[*]};
#do
#    CHECKPOINT=${checkpoint[$i]}
#    EPOCHS=${epochs[$i]}
#    MODEL=${model[$i]}
#    python -u seq2seq_attn.py -checkpoint_dir ${CHECKPOINT} -data_dir ${SOURCE_DIR} -print_every 500\
#    -rnn_size 150 -dropout 0.5 -enc_seq_length 40 -dec_seq_length 100 -max_epochs ${EPOCHS} -seed 100
#    python -u seq2seq_attn_transfer_learning.py -checkpoint_dir ${CHECKPOINT}\
#    -data_dir ${TARGET_DIR} -print_every 300 -seed 100 -model ${MODEL}\
#    -max_epochs 150
#done

cd ../Evaluation

for i in ${!epochs[*]};
do
    CHECKPOINT=${checkpoint[$i]}
    OUTPUT=${output[$i]}
    MODEL=${model[$i]}
    SOURCE=${source[$i]}
#    python Sample.py -data_dir ${SOURCE_DIR}\
#    -output ${SOURCE}\
#    -model ${CHECKPOINT}
    python Sample.py -data_dir ${TARGET_DIR} -output ${OUTPUT} -model ${MODEL}
done