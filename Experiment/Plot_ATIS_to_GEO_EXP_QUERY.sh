#!/usr/bin/env bash

DATA_DIR=../Data/GEO_EXP_QUERY
MODEL_TRANSFER=../Src/checkpoint_atis_geo_exp_query_xx_transfer
MODEL_TARGET=../Src/checkpoint_geo_exp_query_xx
FILENAME=./Plot/Transfer_Learning_ATIS_GEO_EXP_QUERY_ATT

export PYTHONPATH=${PYTHONPATH}:../

cd ../Evaluation

python -u Plot_Transfer_Learning.py -data_dir $DATA_DIR -model_transfer $MODEL_TRANSFER -model_target $MODEL_TARGET\
 -filename $FILENAME -title "Transfer Learning from ATIS Un-anonymised to GeoQuery Un-anonymised (Query-Split)"