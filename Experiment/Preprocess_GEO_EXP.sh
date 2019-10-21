#!/usr/bin/env bash

DIR_NAME=../Data/GEO_EXP

cd ../Preprocess

export PYTHONPATH=${PYTHONPATH}:../

python -u get_vocab_file.py -data_dir $DIR_NAME

python -u generate_input.py -data_dir $DIR_NAME