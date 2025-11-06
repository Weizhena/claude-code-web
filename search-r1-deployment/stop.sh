#!/bin/bash
# 停止所有 Search-R1 服务

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "========================================="
echo "  停止 Search-R1 服务"
echo "========================================="
echo -e "${NC}"
echo ""

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

STOPPED_COUNT=0

# 停止检索服务器
if [ -f logs/retriever.pid ]; then
    RETRIEVER_PID=$(cat logs/retriever.pid)
    if kill -0 $RETRIEVER_PID 2>/dev/null; then
        print_info "停止检索服务器 (PID: $RETRIEVER_PID)..."
        kill $RETRIEVER_PID 2>/dev/null || kill -9 $RETRIEVER_PID 2>/dev/null || true
        sleep 1
        if ! kill -0 $RETRIEVER_PID 2>/dev/null; then
            print_success "检索服务器已停止"
            STOPPED_COUNT=$((STOPPED_COUNT + 1))
        else
            print_error "无法停止检索服务器"
        fi
    else
        print_warning "检索服务器进程不存在 (PID: $RETRIEVER_PID)"
    fi
    rm -f logs/retriever.pid
else
    print_info "未找到检索服务器 PID 文件"
fi

# 停止训练进程
if [ -f logs/training.pid ]; then
    TRAIN_PID=$(cat logs/training.pid)
    if kill -0 $TRAIN_PID 2>/dev/null; then
        print_info "停止训练进程 (PID: $TRAIN_PID)..."
        kill $TRAIN_PID 2>/dev/null || kill -9 $TRAIN_PID 2>/dev/null || true
        sleep 1
        if ! kill -0 $TRAIN_PID 2>/dev/null; then
            print_success "训练进程已停止"
            STOPPED_COUNT=$((STOPPED_COUNT + 1))
        else
            print_error "无法停止训练进程"
        fi
    else
        print_warning "训练进程不存在 (PID: $TRAIN_PID)"
    fi
    rm -f logs/training.pid
else
    print_info "未找到训练 PID 文件"
fi

# 检查端口 8000 是否还被占用
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "端口 8000 仍被占用，尝试终止..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
    if ! lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_success "端口 8000 已释放"
    else
        print_error "无法释放端口 8000"
    fi
fi

# 查找并停止所有相关的 Python 进程
print_info "检查其他相关进程..."
RELATED_PROCESSES=$(ps aux | grep -E "(retrieval_server|verl.trainer)" | grep -v grep | awk '{print $2}' || true)
if [ -n "$RELATED_PROCESSES" ]; then
    print_warning "发现相关进程，正在终止..."
    echo "$RELATED_PROCESSES" | xargs kill -9 2>/dev/null || true
    sleep 1
    print_success "相关进程已清理"
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
if [ $STOPPED_COUNT -gt 0 ]; then
    print_success "已停止 $STOPPED_COUNT 个服务"
else
    print_info "没有运行中的服务"
fi
echo -e "${GREEN}=========================================${NC}"
echo ""
