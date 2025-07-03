#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,9 # Set the GPUs to use
python main.py \
  --config=configs/ve/IXI_128_ncsnpp_continuous.py \
  --eval_folder=eval/IXI \
  --mode='train' \
  --workdir=workdir/IXI/exp0_b64_0702