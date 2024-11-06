#!/bin/bash

CONDA_ENV=
EMAIL=

for n in 8 16 32 64 128; do
    for k in 1 2 4 8; do
        python param_array.py --train-data data_bits/train_${n}_${k}.tsv --val-data data_bits/val_${n}_${k}.tsv --sw-vocab data_bits/vocab_${n}_${k}.tsv --output-dir experiments --model model_${n}_${k} --kernel-size $((8/${k})) --conda ${CONDA_ENV} --email ${EMAIL}
    done
done
