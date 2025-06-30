#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3 # Set the GPUs to use
python main.py \
  --config=configs/ve/IXI_128_ncsnpp_continuous.py \
  --eval_folder=eval/IXI \
  --mode='train' \
  --workdir=workdir/IXI/exp1_b64_0629