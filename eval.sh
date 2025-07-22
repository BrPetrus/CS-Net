#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
echo "Current CUDA setting ${CUDA_VISIBLE_DEVICES}\n"

TIME=$((5 * 60))
echo "Setting time to $(($TIME / 60)) minutes"
ulimit -t $TIME

#source ~/miniforge3/
#/home/xpetrus/miniforge3/bin/conda activate cs2
nice -n 15 python evaluate_tnt.py
