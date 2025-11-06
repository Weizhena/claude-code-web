#!/bin/bash
# Search-R1 训练脚本 - 适配 Qwen3-0.6b

set -e

echo "========================================="
echo "Search-R1 训练 - Qwen3-0.6b"
echo "========================================="
echo ""

# 配置变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU，如果有多个GPU可以设置为 0,1,2,3
export WANDB_MODE=offline      # 离线模式，避免需要 wandb API key

# 数据路径
TRAIN_DATA="../data/sample_train.jsonl"
TEST_DATA="../data/sample_train.jsonl"  # 示例中使用相同数据

# 模型配置
# 选项 1: 使用 HuggingFace 模型（自动下载）
export BASE_MODEL='Qwen/Qwen3-0.6B'

# 选项 2: 使用本地模型路径（如果已下载）
# export BASE_MODEL='../models/Qwen3-0.6B'

# 实验名称
export EXPERIMENT_NAME="search-r1-qwen3-0.6b-demo"
export WAND_PROJECT='Search-R1-Demo'

# 检查检索服务器是否运行
echo "检查检索服务器..."
if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "警告: 检索服务器似乎未运行"
    echo "请先运行: ./start_retriever.sh"
    echo ""
    read -p "是否继续？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据文件不存在: $TRAIN_DATA"
    exit 1
fi

echo "配置信息:"
echo "  模型: $BASE_MODEL"
echo "  实验名称: $EXPERIMENT_NAME"
echo "  训练数据: $TRAIN_DATA"
echo "  测试数据: $TEST_DATA"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# 进入 Search-R1 目录
cd Search-R1

# 使用 GRPO 算法训练（适合小模型）
# 注意：这里的参数已经针对 0.6B 小模型进行了调整
echo "开始训练..."
echo "日志将保存到: ../logs/training.log"
echo ""

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TEST_DATA \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.max_start_length=1024 \
    data.max_obs_length=256 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.total_training_steps=100 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=../checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee ../logs/training.log

echo ""
echo "训练完成！"
echo "检查点保存在: checkpoints/$EXPERIMENT_NAME"
echo "日志文件: logs/training.log"
