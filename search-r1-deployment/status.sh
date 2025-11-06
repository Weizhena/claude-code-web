#!/bin/bash
# 检查 Search-R1 服务状态

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "========================================="
echo "  Search-R1 服务状态"
echo "========================================="
echo -e "${NC}"
echo ""

print_status() {
    if [ "$2" == "running" ]; then
        echo -e "${GREEN}●${NC} $1: ${GREEN}运行中${NC}"
    elif [ "$2" == "stopped" ]; then
        echo -e "${RED}●${NC} $1: ${RED}已停止${NC}"
    else
        echo -e "${YELLOW}●${NC} $1: ${YELLOW}未知${NC}"
    fi
}

# 检查检索服务器
echo "【检索服务器】"
if [ -f logs/retriever.pid ]; then
    RETRIEVER_PID=$(cat logs/retriever.pid)
    if kill -0 $RETRIEVER_PID 2>/dev/null; then
        print_status "检索服务器" "running"
        echo "  PID: $RETRIEVER_PID"
        echo "  URL: http://127.0.0.1:8000"

        # 测试服务器响应
        if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
            echo -e "  健康检查: ${GREEN}通过${NC}"
        else
            echo -e "  健康检查: ${RED}失败${NC}"
        fi
    else
        print_status "检索服务器" "stopped"
        echo "  PID 文件存在但进程不在运行"
    fi
else
    print_status "检索服务器" "stopped"
    echo "  未找到 PID 文件"
fi
echo ""

# 检查训练进程
echo "【训练进程】"
if [ -f logs/training.pid ]; then
    TRAIN_PID=$(cat logs/training.pid)
    if kill -0 $TRAIN_PID 2>/dev/null; then
        print_status "训练进程" "running"
        echo "  PID: $TRAIN_PID"

        # 检查最新日志
        if [ -f logs/training.log ]; then
            LAST_LOG=$(tail -n 1 logs/training.log 2>/dev/null)
            echo "  最新日志: $LAST_LOG"
        fi
    else
        print_status "训练进程" "stopped"
        echo "  PID 文件存在但进程不在运行"
    fi
else
    print_status "训练进程" "stopped"
    echo "  未找到 PID 文件"
fi
echo ""

# 检查 GPU 状态
echo "【GPU 状态】"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "  检测到 GPU: $GPU_COUNT 个"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r idx name mem_used mem_total util; do
        echo "  GPU $idx: $name"
        echo "    内存使用: ${mem_used}MB / ${mem_total}MB"
        echo "    GPU 利用率: ${util}%"
    done
else
    echo "  未检测到 GPU 或 nvidia-smi 不可用"
fi
echo ""

# 检查磁盘空间
echo "【磁盘空间】"
df -h . | tail -n 1 | awk '{print "  可用空间: " $4 " / " $2 " (" $5 " 已使用)"}'
echo ""

# 检查检查点
echo "【训练检查点】"
if [ -d checkpoints ]; then
    CHECKPOINT_COUNT=$(find checkpoints -type d -name "checkpoint-*" 2>/dev/null | wc -l)
    echo "  检查点数量: $CHECKPOINT_COUNT"
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        LATEST_CHECKPOINT=$(ls -td checkpoints/*/checkpoint-* 2>/dev/null | head -n 1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo "  最新检查点: $LATEST_CHECKPOINT"
        fi
    fi
else
    echo "  检查点目录不存在"
fi
echo ""

# 检查日志文件大小
echo "【日志文件】"
if [ -f logs/training.log ]; then
    TRAIN_LOG_SIZE=$(du -h logs/training.log | cut -f1)
    echo "  训练日志: logs/training.log ($TRAIN_LOG_SIZE)"
fi
if [ -f logs/retriever.log ]; then
    RETRIEVER_LOG_SIZE=$(du -h logs/retriever.log | cut -f1)
    echo "  检索日志: logs/retriever.log ($RETRIEVER_LOG_SIZE)"
fi
echo ""

echo -e "${BLUE}=========================================${NC}"
echo ""
echo "命令提示:"
echo "  查看训练日志: tail -f logs/training.log"
echo "  查看检索日志: tail -f logs/retriever.log"
echo "  停止所有服务: ./stop.sh"
echo "  重新启动: ./run.sh"
echo ""
