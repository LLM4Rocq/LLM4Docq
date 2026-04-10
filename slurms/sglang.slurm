#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j_main.out
#SBATCH -e SLURM_Logs/%x_%j_main.err
#SBATCH -D ./
#SBATCH -J multinode-inference-test

#SBATCH --nodes=4
#SBATCH --ntasks=4                    # 1 task par nodes
#SBATCH --ntasks-per-node=1           # Each task will see all GPUs of the node
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # no hyperthreading
#SBATCH --constraint=h100             # h100 gpus
#SBATCH --account=tdm@h100           # account
#SBATCH --qos=qos_gpu_h100-t3

module purge
module load arch/h100 cuda/12.8.0 nccl/2.25.1-1-cuda miniforge/25.9.1

# Activate virtual env

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activating environment on node $SLURM_PROCID"
conda activate sglang

export LD_PRELOAD=$HOME/.conda/envs/sglang/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Define parameters

MODEL="/lustre/fsn1/projects/rech/tdm/ulu88xb/hf_home/hub/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/"

TP_SIZE=16

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running inference"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model: $MODEL"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TP Size: $TP_SIZE"

python3 -m sglang.check_env

# Set NCCL initialization address using the hostname of the head node

HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"

echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"


# Set sglang host ip and disable http proxy

export SGLANG_HOST_IP="${HEAD_NODE}"
unset http_proxy
unset HTTP_PROXY
unset https_proxy
unset HTTPS_PROXY

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $SGLANG_HOST_IP: $HEAD_NODE"

# Launch sglang on each node

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching sglang on all nodes"

srun --output="SLURM_Logs/%x_%j_node%t.out" \
     --error="SLURM_Logs/%x_%j_node%t.err" \
     bash -c 'python3 -m sglang.launch_server --model-path "'"$MODEL"'" --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --grammar-backend "xgrammar" --tp "'"$TP_SIZE"'" --dist-init-addr "'"$NCCL_INIT_ADDR"'" --nnodes "'"$SLURM_NNODES"'" --node-rank "$SLURM_PROCID"' &


# Wait for the server to start

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $NCCL_INIT_ADDR should be ready to accept connection in a few minutes"

sleep 60

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check server health every minute before launching inference"

# This logic (wait/run) should probably be implemented in the python script for inference...

srun --overlap --nodes=1 --nodelist=$HEAD_NODE\
     sh <<'EOF'
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check server health every minute"

until curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/v1/models | grep -q "200"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for server to respond..."
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server running on $HEAD_NODE is ready"

# Keep the script running until the SLURM job times out

export OPENAI_API_KEY=""
cd /lustre/fsn1/projects/rech/tdm/ulu88xb/LLM4Docq
python -m script.compute_docstrings --workers 64

EOF