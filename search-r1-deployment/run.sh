#!/bin/bash
# Search-R1 + Qwen3-0.6b 一键启动脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 清屏
clear

echo -e "${BLUE}"
echo "========================================="
echo "  Search-R1 + Qwen3-0.6b"
echo "  一键启动脚本"
echo "========================================="
echo -e "${NC}"
echo ""

# 函数：打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 函数：清理函数（在退出时调用）
cleanup() {
    echo ""
    print_info "正在清理..."
    if [ -f logs/retriever.pid ]; then
        RETRIEVER_PID=$(cat logs/retriever.pid)
        if kill -0 $RETRIEVER_PID 2>/dev/null; then
            print_info "停止检索服务器 (PID: $RETRIEVER_PID)..."
            kill $RETRIEVER_PID 2>/dev/null || true
        fi
        rm -f logs/retriever.pid
    fi
    print_info "清理完成"
}

# 设置清理陷阱
trap cleanup EXIT INT TERM

# ========================================
# 1. 环境检查
# ========================================
echo -e "${YELLOW}步骤 1/5: 环境检查${NC}"
echo ""

# 检查 Python
if ! command_exists python3; then
    print_error "未找到 python3"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
print_info "Python 版本: $PYTHON_VERSION"

# 检查 CUDA/GPU
if command_exists nvidia-smi; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    print_success "检测到 $GPU_COUNT 个 GPU"
else
    print_warning "未检测到 GPU。此项目需要 GPU 才能运行。"
    print_warning "是否继续（仅用于测试环境检查）？"
    read -p "继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 Search-R1 目录
if [ ! -d "Search-R1" ]; then
    print_error "Search-R1 目录不存在"
    print_info "请先运行 install.sh 进行安装"
    exit 1
fi
print_success "Search-R1 目录存在"

# 检查虚拟环境
if [ -d "venv" ]; then
    print_info "检测到虚拟环境"
    source venv/bin/activate
    print_success "虚拟环境已激活"
elif [[ "$CONDA_DEFAULT_ENV" == "searchr1" ]]; then
    print_success "conda 环境 'searchr1' 已激活"
else
    print_warning "未检测到激活的环境"
    print_info "请先激活虚拟环境："
    print_info "  source venv/bin/activate"
    print_info "或者 conda 环境："
    print_info "  conda activate searchr1"
    exit 1
fi

echo ""

# ========================================
# 2. 检查依赖
# ========================================
echo -e "${YELLOW}步骤 2/5: 检查依赖包${NC}"
echo ""

check_package() {
    if python3 -c "import $1" 2>/dev/null; then
        print_success "$1 已安装"
        return 0
    else
        print_error "$1 未安装"
        return 1
    fi
}

MISSING_PACKAGES=0
for pkg in torch vllm transformers; do
    if ! check_package $pkg; then
        MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
    fi
done

if [ $MISSING_PACKAGES -gt 0 ]; then
    print_error "缺少 $MISSING_PACKAGES 个必需的包"
    print_info "请先运行: ./install.sh"
    exit 1
fi

echo ""

# ========================================
# 3. 准备数据
# ========================================
echo -e "${YELLOW}步骤 3/5: 检查数据文件${NC}"
echo ""

if [ ! -f "data/sample_corpus.jsonl" ]; then
    print_error "语料库文件不存在: data/sample_corpus.jsonl"
    exit 1
fi
print_success "语料库文件存在"

if [ ! -f "data/sample_train.jsonl" ]; then
    print_error "训练数据文件不存在: data/sample_train.jsonl"
    exit 1
fi
print_success "训练数据文件存在"

# 创建必要的目录
mkdir -p logs checkpoints indexes

echo ""

# ========================================
# 4. 启动检索服务器
# ========================================
echo -e "${YELLOW}步骤 4/5: 启动检索服务器${NC}"
echo ""

# 检查端口是否被占用
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "端口 8000 已被占用"
    print_info "尝试终止现有进程..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

print_info "正在启动检索服务器..."
./start_retriever.sh

# 等待服务器就绪
print_info "等待检索服务器就绪..."
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        print_success "检索服务器已就绪"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 1
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_error "检索服务器启动超时"
    print_info "请检查日志: logs/retriever.log"
    exit 1
fi

echo ""
echo ""

# ========================================
# 5. 启动训练
# ========================================
echo -e "${YELLOW}步骤 5/5: 启动训练${NC}"
echo ""

print_info "开始训练 Qwen3-0.6b 模型..."
print_info "这可能需要较长时间，请耐心等待..."
echo ""

# 显示实时日志选项
echo "您可以选择："
echo "  1) 在当前终端查看训练日志"
echo "  2) 后台运行训练（推荐用于长时间训练）"
echo ""
read -p "请选择 (1/2): " -n 1 -r
echo
echo ""

if [[ $REPLY == "2" ]]; then
    print_info "后台运行训练..."
    ./train_qwen3_0.6b.sh > logs/training_output.log 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > logs/training.pid

    print_success "训练已在后台启动 (PID: $TRAIN_PID)"
    echo ""
    print_info "使用以下命令监控训练进度："
    echo -e "${GREEN}  tail -f logs/training.log${NC}"
    echo ""
    print_info "停止训练："
    echo -e "${GREEN}  kill \$(cat logs/training.pid)${NC}"
    echo ""
else
    print_info "在前台运行训练..."
    ./train_qwen3_0.6b.sh
fi

echo ""
echo -e "${GREEN}"
echo "========================================="
echo "  启动完成！"
echo "========================================="
echo -e "${NC}"
echo ""
print_info "检索服务器: http://127.0.0.1:8000"
print_info "检索日志: logs/retriever.log"
print_info "训练日志: logs/training.log"
print_info "检查点: checkpoints/search-r1-qwen3-0.6b-demo/"
echo ""
print_info "按 Ctrl+C 停止所有服务"
echo ""

# 如果是后台运行，等待用户输入
if [[ $REPLY == "2" ]]; then
    print_info "训练正在后台运行..."
    print_info "按 Enter 退出此脚本（不会停止训练）"
    read
fi
