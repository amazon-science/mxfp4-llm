## Example Training Scripts

[`gpt3/`](./gpt3/) folder contains `.sh` scripts, which take 4 command line arguments in the following order: `BW_USE_MXFP4 FW_USE_FP8 FW_EMULATE_FP8 HBS`, that would be treated as environmental variables to control training precision behaviors, specifically,

- `BW_USE_MXFP4` controls backward pass GEMM precision settings
    ```
    0: Use BF16
    1: Use MXFP4 + RHT
    2: Use MXFP4 only
    3: USE MXFP4 + RHT + SR
    4: USE MXFP4 + SR
    ```
- `FW_USE_FP8` controls whether or not to use TransformerEngine's FP8-E4M3 GEMM in the forward pass, with value 1 turned on and 0 (default) turned off to use BF16.

- `FW_EMULATE_FP8` controls whether or not to emulate FP8 GEMM with BF16 upcasting + GEMM. This is useful for running FP8 experiments on non-FP8 hardware such as Ada GPUs, with value 1 turned on and 0 (default) turned off. Note that this is not bit accurate to the hardware FP8 GEMM but is very close ($0.3\%$ difference for i.i.d. Gaussian inputs).

- `HBS` controls the Hadamard block size, defaults to $64$ as used in the paper. Ignored if the MXFP4 backward mode does not use the RHT.

In the script, we launch an instance of the ``pytorch:24.04-py3`` container and mount [`Megatron-LM`](../third_party/Megatron-LM/), [`microxcaling`](../third_party/microxcaling), dataset, and checkpoints. Run the following example command to train GPT3-345M model with BF16 forward + (MXFP4 + RHT + SR) backward recipe:
```bash
bash gpt3/pretrain_gpt345M.sh 3
```

## Multi-node training
Our scripts are configured to be natively compatible with multi-node training, see [gpt3/pretrain_gpt345M_multi_node.slurm](./gpt3/pretrain_gpt345M_multi_node.slurm) as an example on a slurm cluster. Note the `--global-batch-size` config needs to be updated accordingly.

```bash
# sbatch non-interactive
sbatch gpt3/pretrain_gpt345M_multi_node.slurm

# srun interactive
srun gpt3/pretrain_gpt345M.sh 3
```