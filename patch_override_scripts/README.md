## Details of Patches and Overrides

We first modify the OCP MXFP4 datatype to do stochastic rounding. This involves scaling the post MX, pre-quantization input to prevent clipping and then performing stochastic rounding. Then, we apply a blockwise random Hadamard transform (RHT) to the matrix multiplication operands, which allows us to bound the variance of the GEMM output. For more information, see the paper.

We provide two parallel implementations of MXFP4 recipe in this repository:

- [`microxcaling.patch`](./microxcaling.patch) adds support of `dither_scale` rounding mode to `microxcaling/mx/mx_ops.py`, which scales input matrices by $3/4$ before performing stochastic rounding, and scale back by $4/3$ afterwards.
- [`Megatron-LM.patch`](./Megatron-LM.patch) supports training with BF16 forward + MXFP4 backward with major changes to  `megatron/core/tensor_parallel/layers.py`
- [`te1.5/`](./te1.5/) supports FP8 forward + MXFP4 backward by overriding layers defined in `transformer_engine/pytorch/module/*`

## License
This project is licensed under the [`Apache 2.0 license`](https://opensource.org/licenses/Apache-2.0). 

For patches and overrides to `NVIDIA/Megatron-LM`, `microsoft/microxcaling` and `NVIDIA/TransformerEngine`, modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.