#!/bin/bash
#SBATCH --job-name=pretrain_gpt345M
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --time 96:00:00
#SBATCH --mem 1000G
#SBATCH --output=logs/%x-%j.out

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
srun gpt3/pretrain_gpt345M.sh 3