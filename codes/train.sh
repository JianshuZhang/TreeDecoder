#!/bin/bash

# use CUDA
export CUDA_VISIBLE_DEVICES=0
# source ~/.bashrc 
python -u train_wap.py
