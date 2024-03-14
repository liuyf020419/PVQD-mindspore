# !/bin/bash

export USE_CONTEXT=0
export GLOG_v=3
export CUDA_VISIBLE_DEVICES=0

name=diff_gen

python3.8 inference_uncond.py \
    --root_dir ckpt/${name} \
    --monomer_length 100 \
    --decoder_root "./"\
    --step_size 1 \
    --batch_size 1 \
    --model_path ckpt/${name}/last.ckpt \
    --gen_dir "./results/test_diff_gen" \
    --diff_noising_scale 0.1 \