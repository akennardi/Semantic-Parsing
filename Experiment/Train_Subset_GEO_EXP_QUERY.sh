#!/usr/bin/env bash

declare -a subset=(../Data/GEO_EXP_QUERY_10 ../Data/GEO_EXP_QUERY_20 ../Data/GEO_EXP_QUERY_30  ../Data/GEO_EXP_QUERY_40
                   ../Data/GEO_EXP_QUERY_50 ../Data/GEO_EXP_QUERY_60 ../Data/GEO_EXP_QUERY_70
                   ../Data/GEO_EXP_QUERY_80 ../Data/GEO_EXP_QUERY_90 ../Data/GEO_EXP_QUERY)

declare -a checkpoint=(checkpoint_geo_exp_query_10
                       checkpoint_geo_exp_query_20
                       checkpoint_geo_exp_query_30
                       checkpoint_geo_exp_query_40
                       checkpoint_geo_exp_query_50
                       checkpoint_geo_exp_query_60
                       checkpoint_geo_exp_query_70
                       checkpoint_geo_exp_query_80
                       checkpoint_geo_exp_query_90
                       checkpoint_geo_exp_query)

declare -a output=(Output/seq2seq_attention_output_geo_exp_query_10.txt
                   Output/seq2seq_attention_output_geo_exp_query_20.txt
                   Output/seq2seq_attention_output_geo_exp_query_30.txt
                   Output/seq2seq_attention_output_geo_exp_query_40.txt
                   Output/seq2seq_attention_output_geo_exp_query_50.txt
                   Output/seq2seq_attention_output_geo_exp_query_60.txt
                   Output/seq2seq_attention_output_geo_exp_query_70.txt
                   Output/seq2seq_attention_output_geo_exp_query_80.txt
                   Output/seq2seq_attention_output_geo_exp_query_90.txt
                   Output/seq2seq_attention_output_geo_exp_query.txt)

declare -a model=(../Src/checkpoint_geo_exp_query_10/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query_20/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query_30/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query_40/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query_50/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query_60/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query_70/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query_80/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query_90/model_seq2seq_attention
                  ../Src/checkpoint_geo_exp_query/model_seq2seq_attention)

cd ../Src

export PYTHONPATH=${PYTHONPATH}:../

for i in ${!subset[*]};
do
    DATA_DIR=${subset[$i]}
    CHECKPOINT=${checkpoint[$i]}
    python -u seq2seq_attn.py -checkpoint_dir $CHECKPOINT -data_dir $DATA_DIR -print_every 300 -rnn_size 150\
    -dropout 0.5 -enc_seq_length 40 -dec_seq_length 100 -max_epochs 120 -seed 1000
done

cd ../Evaluation

for i in ${!subset[*]};
do
    DATA_DIR=${subset[$i]}
    OUTPUT=${output[$i]}
    MODEL=${model[$i]}
    python Sample.py -data_dir $DATA_DIR -output $OUTPUT -model $MODEL
done