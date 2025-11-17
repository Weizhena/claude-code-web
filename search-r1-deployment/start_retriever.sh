#!/bin/bash
# 启动简化的检索服务器（使用示例语料库）

set -e

echo "========================================="
echo "启动检索服务器"
echo "========================================="
echo ""

# 检查语料库文件是否存在
CORPUS_FILE="./data/sample_corpus.jsonl"
if [ ! -f "$CORPUS_FILE" ]; then
    echo "错误: 语料库文件不存在: $CORPUS_FILE"
    exit 1
fi

# 创建临时索引目录
INDEX_DIR="./indexes/bm25"
mkdir -p "$INDEX_DIR"

echo "语料库文件: $CORPUS_FILE"
echo "索引目录: $INDEX_DIR"
echo "检索器类型: BM25（稀疏检索）"
echo "服务器端口: 8000"
echo ""

# 检查是否已经有检索服务器在运行
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "警告: 端口 8000 已被占用"
    echo "正在尝试终止现有进程..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# 启动检索服务器
echo "正在启动检索服务器..."
cd Search-R1

python3 search_r1/search/retrieval_server.py \
    --index_path "../$INDEX_DIR" \
    --corpus_path "../$CORPUS_FILE" \
    --topk 3 \
    --retriever_name bm25 \
    > ../logs/retriever.log 2>&1 &

RETRIEVER_PID=$!
echo $RETRIEVER_PID > ../logs/retriever.pid

cd ..

echo ""
echo "检索服务器已启动！"
echo "  PID: $RETRIEVER_PID"
echo "  URL: http://127.0.0.1:8000/retrieve"
echo "  日志: logs/retriever.log"
echo ""
echo "等待服务器就绪..."
sleep 5

# 测试检索服务器
echo "测试检索服务器..."
# 使用 curl 内置重试机制等待服务就绪
if curl --silent --fail --retry 10 --retry-delay 1 --retry-connrefused \
    http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✓ 检索服务器运行正常 (健康检查通过)"
else
    echo "⚠ 健康检查端点不可用，尝试检索端点..."
    if curl -s -X POST http://127.0.0.1:8000/retrieve \
        -H "Content-Type: application/json" \
        -d '{"query": "test"}' > /dev/null 2>&1; then
        echo "✓ 检索服务器运行正常 (检索端点可用)"
    else
        echo "✗ 检索服务器可能未正确启动，请检查日志: logs/retriever.log"
    fi
fi

echo ""
echo "使用 'tail -f logs/retriever.log' 查看日志"
echo "使用 'kill $(cat logs/retriever.pid)' 停止服务器"
