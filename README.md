# PVQD
Protein structure Vector Quantization Diffusion model.

PVQD is a method based on deep learning for protein structure design and prediction. Here we published the source code and the demos for PVQD. This code is developed on Mindspore framework.

## Install Dependencies
First, please install the depandencies of PVQD
```
conda create -n PVQD python=3.8
conda activate PVQD

pip install -r ./install/requirement.txt
pip install ./install/mindsponge_gpu-1.0.0rc2-py3-none-any.whl
```
Then, download the weight from https://biocomp.ustc.edu.cn/servers/downloads/PVQD_mindspore_ckpt.tar.gz, and extract the archive files `PVQD_mindspore_ckpt.tar.gz` into the directory containing the project files (the same directory where the corresponding .sh files are located).


## Quick start

Three scripts related to the manuscript are available:
* `sample_from_fa.sh` - structure prediction
* `sample_uncond.sh` - unconditional structure generation
* `pdbf_sample.sh` - encode and decode protein structure to generate an latent space array

The results of demos are saved in `results`.

