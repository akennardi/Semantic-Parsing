#!/usr/bin/env bash

SOURCE_DIR=../Data/GEO_EXP_Query

declare -a subset=(../Data/GEO_EXP_Query_10 ../Data/GEO_EXP_Query_20 ../Data/GEO_EXP_Query_30  ../Data/GEO_EXP_Query_40
                   ../Data/GEO_EXP_Query_50 ../Data/GEO_EXP_Query_60 ../Data/GEO_EXP_Query_70
                   ../Data/GEO_EXP_Query_80 ../Data/GEO_EXP_Query_90)

declare -a size=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

cd ../Preprocess

export PYTHONPATH=${PYTHONPATH}:../

for i in ${!subset[*]};
do
    TARGET_DIR=${subset[$i]}
    SIZE=${size[$i]}
    python -u split_data_bucket.py -source_dir $SOURCE_DIR -target_dir $TARGET_DIR -subset_size $SIZE

    python -u get_vocab_file.py -data_dir $TARGET_DIR

    python -u generate_input.py -data_dir $TARGET_DIR

done

