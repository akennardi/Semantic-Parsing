#!/usr/bin/env bash

declare -a subset=(../Data/GEO_10 ../Data/GEO_20 ../Data/GEO_30  ../Data/GEO_40 ../Data/GEO_50
                   ../Data/GEO_60 ../Data/GEO_70 ../Data/Geo_80  ../Data/GEO_90 ../Data/GEO)

declare -a checkpoint=(checkpoint_geo_10
                       checkpoint_geo_20
                       checkpoint_geo_30
                       checkpoint_geo_40
                       checkpoint_geo_50
                       checkpoint_geo_60
                       checkpoint_geo_70
                       checkpoint_geo_80
                       checkpoint_geo_90
                       checkpoint_geo)

declare -a output=(Output/seq2seq_attention_output_geo_10.txt
                   Output/seq2seq_attention_output_geo_20.txt
                   Output/seq2seq_attention_output_geo_30.txt
                   Output/seq2seq_attention_output_geo_40.txt
                   Output/seq2seq_attention_output_geo_50.txt
                   Output/seq2seq_attention_output_geo_60.txt
                   Output/seq2seq_attention_output_geo_70.txt
                   Output/seq2seq_attention_output_geo_80.txt
                   Output/seq2seq_attention_output_geo_90.txt
                   Output/seq2seq_attention_output_geo.txt)

declare -a model=(../Model/checkpoint_geo_10/model_seq2seq_attention
                  ../Model/checkpoint_geo_20/model_seq2seq_attention
                  ../Model/checkpoint_geo_30/model_seq2seq_attention
                  ../Model/checkpoint_geo_40/model_seq2seq_attention
                  ../Model/checkpoint_geo_50/model_seq2seq_attention
                  ../Model/checkpoint_geo_60/model_seq2seq_attention
                  ../Model/checkpoint_geo_70/model_seq2seq_attention
                  ../Model/checkpoint_geo_80/model_seq2seq_attention
                  ../Model/checkpoint_geo_90/model_seq2seq_attention
                  ../Model/checkpoint_geo/model_seq2seq_attention)

cd ../Model

export PYTHONPATH=${PYTHONPATH}:../

for i in ${!subset[*]};
do
    DATA_DIR=${subset[$i]}
    CHECKPOINT=${checkpoint[$i]}
    python -u seq2seq_attn.py -checkpoint_dir $CHECKPOINT -data_dir $DATA_DIR -print_every 300 -rnn_size 150\
    -dropout 0.5 -enc_seq_length 40 -dec_seq_length 100 -max_epochs 90 -seed 1000
done

cd ../Evaluation

for i in ${!subset[*]};
do
    DATA_DIR=${subset[$i]}
    OUTPUT=${output[$i]}
    MODEL=${model[$i]}
    python Sample.py -data_dir $DATA_DIR -output $OUTPUT -model $MODEL
done