#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH -o logs/train_multi_%j.out
#SBATCH -e logs/train_multi_%j.err
#SBATCH --time=00:30:00

mkdir -p logs
module load course/cme213/nvhpc/24.1

echo "=== Multi-GPU Training (4 GPUs, MPI) ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mpirun -np 4 ./build_mpi/train_mpi inp.txt
