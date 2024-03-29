#!/usr/bin/env bash

SOURCE_DIR=../Data/ATIS
declare -a subset=(../Data/GEO_10 ../Data/GEO_20 ../Data/GEO_30  ../Data/GEO_40 ../Data/GEO_50
                   ../Data/GEO_60 ../Data/GEO_70 ../Data/Geo_80  ../Data/GEO_90 ../Data/GEO)

declare -a checkpoint=(checkpoint_atis_geo_10_transfer
                       checkpoint_atis_geo_20_transfer
                       checkpoint_atis_geo_30_transfer
                       checkpoint_atis_geo_40_transfer
                       checkpoint_atis_geo_50_transfer
                       checkpoint_atis_geo_60_transfer
                       checkpoint_atis_geo_70_transfer
                       checkpoint_atis_geo_80_transfer
                       checkpoint_atis_geo_90_transfer
                       checkpoint_atis_geo_transfer)

declare -a output=(Output/seq2seq_attention_output_atis_geo_10_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_20_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_30_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_40_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_50_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_60_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_70_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_80_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_90_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_transfer.txt)

declare -a model=(../Src/checkpoint_atis_geo_10_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_20_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_30_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_40_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_50_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_60_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_70_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_80_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_90_transfer/model_seq2seq_attention
                  ../Src/checkpoint_atis_geo_transfer/model_seq2seq_attention)

cd ../Src

export PYTHONPATH=${PYTHONPATH}:../

python -u seq2seq_attn.py -checkpoint_dir checkpoint_atis_transfer -data_dir $SOURCE_DIR -print_every 500\
 -rnn_size 150 -dropout 0.5 -enc_seq_length 40 -dec_seq_length 100 -max_epochs 15 -seed 100

for i in ${!subset[*]};
do
    TARGET_DIR=${subset[$i]}
    CHECKPOINT=${checkpoint[$i]}
    python -u seq2seq_attn_transfer_learning.py -checkpoint_dir $CHECKPOINT\
    -data_dir $TARGET_DIR -print_every 300 -seed 100 -model ../Src/checkpoint_atis_transfer/model_seq2seq_attention\
     -max_epochs 150
done

cd ../Evaluation

python Sample.py -data_dir $SOURCE_DIR\
 -output Output/seq2seq_attention_output_atis_transfer.txt\
 -model ../Src/checkpoint_atis_transfer/model_seq2seq_attention

for i in ${!subset[*]};
do
    TARGET_DIR=${subset[$i]}
    OUTPUT=${output[$i]}
    MODEL=${model[$i]}
    python Sample.py -data_dir $TARGET_DIR -output $OUTPUT -model $MODEL
done

