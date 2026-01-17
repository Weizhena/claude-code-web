#!/usr/bin/env bash
set -eo pipefail
source /data/wjc2025/miniconda3/etc/profile.d/conda.sh
conda activate LLM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

/data/wjc2025/miniconda3/envs/LLM/bin/python -m torch.distributed.run \
  --nproc_per_node=6 --master_port 29504 \
  /data/wjc2025/method/method/mem_alpha/personamem_grpo_trainer.py \
  --data-dir /data/wjc2025/method/method/mem_alpha/data/personamem \
  --model /data/wjc2025/method/method/mem_alpha/model/qwen3-4b \
  --lora-adapter /data/wjc2025/method/method/mem_alpha/output/personamem_sft_lora/lora_adapter \
  --deepspeed-config /data/wjc2025/method/method/mem_alpha/deepspeed_grpo.json \
  --load-in-4bit --batch-size 2 \
  --output-dir /data/wjc2025/method/method/mem_alpha/output/personamem_grpo
