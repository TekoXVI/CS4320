#!/bin/bash

# ./split_data.py --data-file TMNIST_Data.csv --test-ratio 0.20
# ./split_data.py --data-file TMNIST_Data-train.csv --test-ratio 0.30 --train-file TMNIST-train-fit.csv --test-file TMNIST-validation.csv

./cnn_classification.py cnn-fit --train-file TMNIST-train-fit.csv --model-file big-a.h5
./cnn_classification.py score --train-file TMNIST-train-fit.csv --test-file TMNIST-validation.csv --show-test 1  --model-file big-a.h5
# 
# ######
# ./cnn_classification.py score --train-file TMNIST-train-fit.csv --test-file TMNIST-test.csv --show-test 1  --model-file big-a.h5
# ######
