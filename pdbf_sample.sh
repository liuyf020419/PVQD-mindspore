#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export MULTI_CHAIN=0
export GLOG_v=3

testlist=./results/test_pdb/testlist.txt
pdbroot=./results/test_pdb/pdb_dir
gen_dir=./results/test_pdb/
name=structure_vq

python3.8 inference_pdbf_data.py \
    --root_dir ckpt/${name} \
    --test_list ${testlist} \
    --gen_dir ${gen_dir} \
    --max_sample_num 50 \
    --pdb_root ${pdbroot} \
    --write_pdbfile \
    --batch_size 1 \
    --num_workers 1 \
    --return_structure \
    --model_path ckpt/${name}/last.ckpt