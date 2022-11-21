#! /bin/bash

export AWS_ACCESS_KEY_ID=laibo
export AWS_SECRET_ACCESS_KEY=lettherebelight
export MLFLOW_S3_ENDPOINT_URL=http://192.168.11.249:9000

python train_pl.py > train_pl.log 2>&1 &
