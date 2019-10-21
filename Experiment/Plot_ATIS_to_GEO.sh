#!/usr/bin/env bash

DATA_DIR=../Data/GEO
MODEL_TRANSFER=../Model/checkpoint_atis_geo_xx_transfer
MODEL_TARGET=../Model/checkpoint_geo_xx
FILENAME=./Plot/Transfer_Learning_ATIS_GEO_ATT

export PYTHONPATH=${PYTHONPATH}:../

cd ../Evaluation

python -u Plot_Transfer_Learning.py -data_dir $DATA_DIR -model_transfer $MODEL_TRANSFER -model_target $MODEL_TARGET\
 -filename $FILENAME -title "Transfer Learning from ATIS to GeoQuery"