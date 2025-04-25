#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
set -o pipefail

BW_USE_MXFP4=$1
FW_USE_FP8=${2:-0}
FW_EMULATE_FP8=${3:-0}
HBS=${4:-64}

ROOT_DIR=$(dirname "$(pwd)")
MEGATRON_SCRIPT_PATH=$ROOT_DIR/third_party/Megatron-LM
MICROXCALING_PATH=$ROOT_DIR/third_party/microxcaling
DATA_PATH=$ROOT_DIR/examples_datasets/gpt2
CHECKPOINT_PATH=$ROOT_DIR/ckpts_gpt345M
TE_OVERRIDE_PATH=$ROOT_DIR/patch_override_scripts/te1.5

ulimit -n 65535
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
if [ -v SLURM_NNODES ]; then
    # SLURM runs, single or multi-node
    IPS=""
    for h in $(scontrol show hostname); do
        IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
    TS_PATH=/workspace/checkpoints/gpt_6b7_BW${BW_USE_MXFP4}_FP8FW${FW_USE_FP8}_FP8EMU${FW_EMULATE_FP8}_JOB${SLURM_JOB_ID}_$(date '+%Y%m%d%H%M%S')
else
    # Single-node, non-SLURM runs
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    TS_PATH=/workspace/checkpoints/gpt_6b7_BW${BW_USE_MXFP4}_FP8FW${FW_USE_FP8}_FP8EMU${FW_EMULATE_FP8}_$(date '+%Y%m%d%H%M%S')
fi

# export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=54321

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

IMAGE_URI=nvcr.io/nvidia/pytorch:24.04-py3
DOCKER_NAME=nvidia-pytorch-24.04
docker stop $DOCKER_NAME
docker pull $IMAGE_URI
docker images
docker run --name $DOCKER_NAME --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -d \
    --net=host --uts=host \
    -e NCCL_SOCKET_IFNAME="^lo,docker0" \
    -e RDMAV_FORK_SAFE=1 \
    -e GPU_NUM_DEVICES=8 \
    -e FI_EFA_USE_DEVICE_RDMA=1  \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e BW_USE_MXFP4=$BW_USE_MXFP4 \
    -e FW_EMULATE_FP8=$FW_EMULATE_FP8 \
    -e HBS=$HBS \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size=512G \
    -v $MEGATRON_SCRIPT_PATH:/workspace/megatron_lm \
    -v $MICROXCALING_PATH:/workspace/microxcaling \
    -v $DATA_PATH:/workspace/datasets \
    -v $CHECKPOINT_PATH:/workspace/checkpoints \
    -v $TE_OVERRIDE_PATH:/workspace/te_override \
    $IMAGE_URI

GPT_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --lr 0.00012 \
    --train-iters 40000 \
    --lr-decay-iters 40000 \
    --lr-decay-style cosine \
    --min-lr 0.000012 \
    --weight-decay 0.1 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --use-distributed-optimizer \
    --tensor-model-parallel-size 8 \
    --no-gradient-accumulation-fusion
"

DOCKER_VOCAB_FILE=/workspace/datasets/gpt2-vocab.json
DOCKER_MERGE_FILE=/workspace/datasets/gpt2-merges.txt
DOCKER_DATA_PATH=/workspace/datasets/my-gpt2_text_document

DATA_ARGS="
    --data-path $DOCKER_DATA_PATH \
    --vocab-file $DOCKER_VOCAB_FILE \
    --merge-file $DOCKER_MERGE_FILE \
    --data-cache-path /workspace/datasets/index-cache \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 5000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --timing-log-level 2
"

docker exec $DOCKER_NAME bash -c "cd /workspace/microxcaling && pip install ."
if [ $FW_USE_FP8 -eq 1 ]; then
    FP8_ARGS="
        --fp8-e4m3 \
        --transformer-impl transformer_engine
    "
    docker exec $DOCKER_NAME bash -c "cp /workspace/te_override/linear.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/module/linear.py"
    docker exec $DOCKER_NAME bash -c "cp /workspace/te_override/layernorm_linear.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/module/layernorm_linear.py"
    docker exec $DOCKER_NAME bash -c "cp /workspace/te_override/layernorm_mlp.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/module/layernorm_mlp.py"
    docker exec $DOCKER_NAME bash -c "cp /workspace/te_override/fp8.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/fp8.py"
else
    FP8_ARGS=""
fi

docker exec $DOCKER_NAME torchrun $DISTRIBUTED_ARGS /workspace/megatron_lm/pretrain_gpt.py \
       $GPT_ARGS \
       $DATA_ARGS \
       $OUTPUT_ARGS \
       $FP8_ARGS \
       --save $TS_PATH \
       --load $TS_PATH \
       --tensorboard-dir $TS_PATH/tb &
wait %1