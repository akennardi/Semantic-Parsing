#!/usr/bin/env bash

SOURCE_DIR=../Data/ATIS_EXP
declare -a subset=(../Data/GEO_EXP_10 ../Data/GEO_EXP_20 ../Data/GEO_EXP_30  ../Data/GEO_EXP_40 ../Data/GEO_EXP_50
                   ../Data/GEO_EXP_60 ../Data/GEO_EXP_70 ../Data/Geo_EXP_80  ../Data/GEO_EXP_90 ../Data/GEO_EXP)

declare -a checkpoint=(checkpoint_atis_geo_exp_10_transfer
                       checkpoint_atis_geo_exp_20_transfer
                       checkpoint_atis_geo_exp_30_transfer
                       checkpoint_atis_geo_exp_40_transfer
                       checkpoint_atis_geo_exp_50_transfer
                       checkpoint_atis_geo_exp_60_transfer
                       checkpoint_atis_geo_exp_70_transfer
                       checkpoint_atis_geo_exp_80_transfer
                       checkpoint_atis_geo_exp_90_transfer
                       checkpoint_atis_geo_exp_transfer)

declare -a output=(Output/seq2seq_attention_output_atis_geo_exp_10_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_20_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_30_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_40_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_50_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_60_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_70_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_80_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_90_transfer.txt
                   Output/seq2seq_attention_output_atis_geo_exp_transfer.txt)

declare -a model=(../Model/checkpoint_atis_geo_exp_10_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_20_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_30_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_40_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_50_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_60_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_70_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_80_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_90_transfer/model_seq2seq_attention
                  ../Model/checkpoint_atis_geo_exp_transfer/model_seq2seq_attention)

cd ../Model

export PYTHONPATH=${PYTHONPATH}:../

python -u seq2seq_attn.py -checkpoint_dir checkpoint_atis_exp_transfer -data_dir $SOURCE_DIR -print_every 500\
 -rnn_size 150 -dropout 0.5 -enc_seq_length 40 -dec_seq_length 100 -max_epochs 15 -seed 1000

for i in ${!subset[*]};
do
    TARGET_DIR=${subset[$i]}
    CHECKPOINT=${checkpoint[$i]}
    python -u seq2seq_attn_transfer_learning.py -checkpoint_dir $CHECKPOINT\
    -data_dir $TARGET_DIR -print_every 300 -seed 1000\
     -model ../Model/checkpoint_atis_exp_transfer/model_seq2seq_attention -max_epochs 180
done

cd ../Evaluation

python Sample.py -data_dir $SOURCE_DIR\
 -output Output/seq2seq_attention_output_atis_exp_transfer.txt\
 -model ../Model/checkpoint_atis_exp_transfer/model_seq2seq_attention

for i in ${!subset[*]};
do
    TARGET_DIR=${subset[$i]}
    OUTPUT=${output[$i]}
    MODEL=${model[$i]}
    python Sample.py -data_dir $TARGET_DIR -output $OUTPUT -model $MODEL
done