#!/bin/bash

export USE_CONTEXT=1
export CUDA_VISIBLE_DEVICES=0
export GLOG_v=3

name=structure_prediction
fasta_f=./results/test_inference_from_fa/cameo2022.fa

python3.8 inference_from_fa.py \
    --decoder_root "./"\
    --root_dir ckpt/${name} \
    --fasta_f ${fasta_f} \
    --step_size 1 \
    --gen_dir "./results/test_inference_from_fa/"\
    --batch_size 2 \
    --model_path ckpt/${name}/inference_fa_crynmr.ckpt \
    --diff_noising_scale 0.1 \