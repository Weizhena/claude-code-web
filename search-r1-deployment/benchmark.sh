#!/bin/bash
# Search-R1 系统环境检测和性能基准测试脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================="
echo "Search-R1 系统环境检测和性能基准测试"
echo "========================================="
echo ""

# 1. 检查 nvidia-smi 是否可用
echo -e "${BLUE}=== GPU 检测 ===${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi 未找到${NC}"
    echo "NVIDIA 驱动可能未安装"
    exit 1
fi

echo -e "${GREEN}✓ nvidia-smi 已找到${NC}"
echo ""

# 2. 显示 GPU 基本信息
echo -e "${BLUE}--- GPU 信息 ---${NC}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free \
    --format=csv,noheader | while IFS=',' read -r index name driver memory_total memory_free; do
    echo "GPU $index: $name"
    echo "  驱动版本: $driver"
    echo "  总内存: $memory_total"
    echo "  可用内存: $memory_free"
done
echo ""

# 3. 检查 CUDA 版本
echo -e "${BLUE}--- CUDA 版本 ---${NC}"
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | grep -oP 'CUDA Version: \K[0-9.]+' || echo "未显示")
echo "CUDA 驱动版本: $CUDA_VERSION"

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | grep -oP 'release \K[0-9.]+')
    echo "CUDA Toolkit 版本: $NVCC_VERSION"
else
    echo "CUDA Toolkit: 未安装或不在 PATH 中"
fi
echo ""

# 4. GPU 数量和使用情况
echo -e "${BLUE}--- GPU 状态 ---${NC}"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "GPU 数量: $GPU_COUNT"
echo ""
echo "GPU 使用情况:"
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv,noheader | while IFS=',' read -r index gpu_util mem_util temp power; do
    echo "  GPU $index: GPU利用率=$gpu_util, 内存利用率=$mem_util, 温度=$temp, 功耗=$power"
done
echo ""

# 5. Python 环境检测
echo -e "${BLUE}=== Python 环境 ===${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
    echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python3 未找到${NC}"
    exit 1
fi
echo ""

# 6. PyTorch GPU 验证
echo -e "${BLUE}=== PyTorch GPU 验证 ===${NC}"
python3 << 'EOF'
import sys

try:
    import torch
    print(f"✓ PyTorch 版本: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA 可用: 是")
        print(f"  CUDA 版本 (PyTorch): {torch.version.cuda}")
        print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    计算能力: {props.major}.{props.minor}")
            print(f"    总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"    多处理器数量: {props.multi_processor_count}")
    else:
        print("✗ CUDA 不可用")
        sys.exit(1)

except ImportError as e:
    print(f"✗ PyTorch 未安装: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ 错误: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch GPU 验证通过${NC}"
else
    echo -e "${RED}✗ PyTorch GPU 验证失败${NC}"
    exit 1
fi
echo ""

# 7. 检查其他依赖
echo -e "${BLUE}=== 依赖检查 ===${NC}"

check_python_package() {
    local package=$1
    if python3 -c "import $package" 2>/dev/null; then
        local version=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null || echo "未知")
        echo -e "${GREEN}✓ $package: $version${NC}"
    else
        echo -e "${YELLOW}⚠ $package: 未安装${NC}"
    fi
}

check_python_package "transformers"
check_python_package "vllm"
check_python_package "fastapi"
check_python_package "uvicorn"
check_python_package "wandb"
echo ""

# 8. 简单的性能测试
echo -e "${BLUE}=== GPU 性能测试 ===${NC}"
echo "运行简单的矩阵乘法测试..."

python3 << 'EOF'
import torch
import time

if not torch.cuda.is_available():
    print("CUDA 不可用，跳过性能测试")
    exit(0)

device = torch.device("cuda:0")
print(f"使用设备: {torch.cuda.get_device_name(0)}")

# 预热
_ = torch.randn(1000, 1000, device=device) @ torch.randn(1000, 1000, device=device)
torch.cuda.synchronize()

# 性能测试
sizes = [1000, 2000, 4000]
for size in sizes:
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start = time.time()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - start

    gflops = (2 * size ** 3) / elapsed / 1e9
    print(f"矩阵大小 {size}x{size}: {elapsed*1000:.2f} ms, {gflops:.2f} GFLOPS")

print("\n✓ 性能测试完成")
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GPU 性能测试完成${NC}"
else
    echo -e "${YELLOW}⚠ GPU 性能测试失败${NC}"
fi
echo ""

# 9. 磁盘空间检查
echo -e "${BLUE}=== 磁盘空间 ===${NC}"
DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
DISK_AVAIL=$(df -h . | tail -1 | awk '{print $4}')

echo "当前目录: $(pwd)"
echo "可用空间: $DISK_AVAIL"
echo "使用率: $DISK_USAGE%"

if [ "$DISK_USAGE" -gt 90 ]; then
    echo -e "${RED}⚠ 警告: 磁盘使用率超过 90%${NC}"
elif [ "$DISK_USAGE" -gt 80 ]; then
    echo -e "${YELLOW}⚠ 注意: 磁盘使用率超过 80%${NC}"
else
    echo -e "${GREEN}✓ 磁盘空间充足${NC}"
fi
echo ""

# 10. 内存检查
echo -e "${BLUE}=== 系统内存 ===${NC}"
if command -v free &> /dev/null; then
    free -h | grep -E "Mem|Swap"

    TOTAL_MEM=$(free -g | grep Mem | awk '{print $2}')
    AVAIL_MEM=$(free -g | grep Mem | awk '{print $7}')

    echo ""
    echo "总内存: ${TOTAL_MEM}GB"
    echo "可用内存: ${AVAIL_MEM}GB"

    if [ "$AVAIL_MEM" -lt 8 ]; then
        echo -e "${YELLOW}⚠ 警告: 可用内存少于 8GB${NC}"
    else
        echo -e "${GREEN}✓ 内存充足${NC}"
    fi
else
    echo "free 命令不可用，跳过内存检查"
fi
echo ""

# 完成
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}环境检测和性能测试完成！${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "系统准备就绪，可以运行 Search-R1 训练任务"
