#!/bin/bash
# Search-R1 + Qwen3-0.6b 自动安装脚本

set -e

echo "========================================="
echo "Search-R1 + Qwen3-0.6b 安装脚本"
echo "========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 CUDA
echo -e "${YELLOW}检查 CUDA 环境...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}警告: 未检测到 CUDA/GPU。此项目需要 GPU 才能运行。${NC}"
    echo "请在具有 NVIDIA GPU 的机器上运行此脚本。"
    read -p "是否继续安装（仅用于测试）？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ 检测到 CUDA 环境${NC}"
    nvidia-smi | head -n 10
fi

# 检查 Python 版本
echo ""
echo -e "${YELLOW}检查 Python 环境...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "检测到 Python 版本: $PYTHON_VERSION"

# 创建虚拟环境（如果没有 conda）
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}未检测到 conda，使用 venv 创建虚拟环境...${NC}"

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✓ 虚拟环境已创建${NC}"
    else
        echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
    fi

    source venv/bin/activate
    echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
else
    echo -e "${YELLOW}检测到 conda，使用 conda 创建环境...${NC}"

    # 检查环境是否存在
    if conda env list | grep -q "searchr1"; then
        echo -e "${GREEN}✓ conda 环境 'searchr1' 已存在${NC}"
    else
        conda create -n searchr1 python=3.9 -y
        echo -e "${GREEN}✓ conda 环境 'searchr1' 已创建${NC}"
    fi

    echo "请手动激活 conda 环境："
    echo "  conda activate searchr1"
    echo "然后重新运行此脚本。"
    exit 0
fi

# 更新 pip
echo ""
echo -e "${YELLOW}更新 pip...${NC}"
pip install --upgrade pip

# 安装 PyTorch
echo ""
echo -e "${YELLOW}安装 PyTorch 2.4.0 (CUDA 12.1)...${NC}"
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 vLLM
echo ""
echo -e "${YELLOW}安装 vLLM 0.6.3...${NC}"
pip install vllm==0.6.3

# 进入 Search-R1 目录并安装
echo ""
echo -e "${YELLOW}安装 Search-R1 依赖...${NC}"
cd Search-R1
pip install -e .

# 安装 Flash Attention 2
echo ""
echo -e "${YELLOW}安装 Flash Attention 2...${NC}"
pip install flash-attn --no-build-isolation

# 安装其他工具
echo ""
echo -e "${YELLOW}安装其他依赖...${NC}"
pip install wandb uvicorn fastapi

# 完成
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}安装完成！${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "下一步："
echo "1. 配置 Qwen3-0.6b 模型路径"
echo "2. 准备数据集和语料库"
echo "3. 运行 ./run.sh 启动训练"
echo ""
echo "如果使用 conda，请先激活环境："
echo "  conda activate searchr1"
