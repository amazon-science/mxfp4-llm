# [Training LLMs with MXFP4](https://arxiv.org/abs/2502.20586)

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2405.03637&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2502.20586)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

This repo contains official implementation for Training LLMs with MXFP4. Our MXFP4 training recipe achieves near-lossless training by computing unbiased gradient estimates (with stochastic rounding and random Hadamard transformation) using MXFP4-accelerated GEMMs. This allows us compute the backward pass in MXFP4, which constitutes $>1/2$ of the FLOPs during training.

We support training with [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM/tree/main) and [`NVIDIA/TransformerEngine`](https://github.com/NVIDIA/TransformerEngine/tree/main). Due to lack of MXFP4 hardware supports (Blackwell GPUs), we use [`microsoft/microxcaling`](https://github.com/microsoft/microxcaling) to perform emulation of MXFP4 GEMMS [(OCP MX specification)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).


## Requirements
We recommend using [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) with released tag ``pytorch:24.04-py3``
```bash
docker pull nvcr.io/nvidia/pytorch:24.04-py3
```
We support MXFP4 backward passes with both BF16 and FP8 forward passes, leveraging TransformerEngine for the latter. Currently, we only supported FP8 + MXFP4 training with [TransformerEngine-Version('1.5.0+6a9edc3')](https://github.com/NVIDIA/TransformerEngine/tree/release_v1.5), which comes pre-installed in the ``pytorch:24.04-py3`` container.

## Datasets
We used the `GPT2BPETokenizer` preprocessed Wikipedia dataset (around 3.28 billion tokens). Please follow [AWS-Neuron-Examples-Megatron-LM-GPT](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/frameworks/torch/torch-neuronx/tutorials/training/megatron_lm_gpt.html#download-preprocessed-wikipedia-dataset) to 
download from s3. 
```
export DATA_DIR=./examples_datasets/gpt2
mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.bin .  --no-sign-request
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.idx .  --no-sign-request
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/license.txt .  --no-sign-request
```
Alternatively, you can also prepare custom dataset with [`Megatron-LM/tools/preprocess_data.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py) to build megatron-compatible mmap format `.bin` & `.idx`  from scratch, as illustrated in [preparing-wikipedia-dataset-from-scratch](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/frameworks/torch/torch-neuronx/tutorials/training/megatron_lm_gpt.html#preparing-wikipedia-dataset-from-scratch).


## Usage

### Prerequisites
Clone the repository and submodule with the following command:
```
git clone https://github.com/amazon-science/mxfp4-llm
cd mxfp4-llm
git submodule update --init --recursive
```

### Apply patch to Megatron-LM & microxcaling
We made changes to the official `NVIDIA/Megatron-LM`-v0.2.0 and `microsoft/microxcaling`-v1.1.1.dev0, and packaged into patches. Apply these patches to `third_party/*` with 
```bash
cd third_party/Megatron-LM
git apply ../../patch_override_scripts/Megatron-LM.patch
cd ../microxcaling
git apply ../../patch_override_scripts/microxcaling.patch
cd ../../scripts
```
A detailed description of the changes can be found in [`patch_override_scripts`](./patch_override_scripts/). Check our paper for more information.

### Pretrain GPT
We provide scripts to train GPT3-345M, 1.3B, and 6.7B parameter models in [`scripts/gpt3`](./scripts/gpt3/), with a guideline on configurable precision options at [`scripts`](./scripts/). This code is well-tested on Ada (A100, L40S) and Hopper (H100) GPUs. 

### Planned features
- Migrate to newer third_party package versions, e.g. Megatron-v0.13.0rc0 and TransformerEngine-v1.14.0, in order to support `--tp-comm-overlap` for the forward pass GEMM
- Add LLaMA pretraining scripts
- Utilize MXFP4 GEMM support on Blackwell GPUs

## Contributing

This project welcomes contributions and suggestions, see [CONTRIBUTING.md](./CONTRIBUTING.md) for details. This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact opensource-codeofconduct@amazon.com with any additional questions or comments.

## License
This project is licensed under the [`Apache 2.0 license`](https://opensource.org/licenses/Apache-2.0). 

## Cite us

If you find our works helpful in your research, please consider citing the following paper:
```
@inproceedings{
tseng2025training,
title={Training {LLM}s with {MXFP}4},
author={Albert Tseng and Tao Yu and Youngsuk Park},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=a8z5Q0WSPL}
}
```