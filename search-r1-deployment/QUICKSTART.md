# 🚀 Search-R1 + Qwen3-0.6b 快速开始指南

5 分钟内启动并运行！

## ⚡ 超快速开始（3 步）

```bash
# 1. 安装依赖
./install.sh

# 2. 激活环境
source venv/bin/activate  # 或: conda activate searchr1

# 3. 一键启动
./run.sh
```

就这么简单！🎉

## 📦 需要什么

- **GPU**: NVIDIA GPU (12GB+ VRAM 推荐)
- **系统**: Linux (Ubuntu 20.04/22.04)
- **时间**: 约 15-30 分钟安装 + 训练时间

## 🎯 详细步骤

### 步骤 1: 克隆或解压此项目

如果你已经在这个目录，跳过此步骤。

```bash
cd search-r1-deployment
```

### 步骤 2: 运行安装脚本

```bash
./install.sh
```

这会：
- ✅ 检查 CUDA/GPU
- ✅ 创建虚拟环境
- ✅ 安装 PyTorch、vLLM、Flash Attention 等
- ✅ 安装 Search-R1 框架

**预计时间**: 10-20 分钟（取决于网速）

### 步骤 3: 激活环境

**如果使用 venv**:
```bash
source venv/bin/activate
```

**如果使用 conda**:
```bash
conda activate searchr1
```

### 步骤 4: 启动训练

```bash
./run.sh
```

这会自动：
1. 检查环境和依赖 ✓
2. 启动 BM25 检索服务器 ✓
3. 等待服务器就绪 ✓
4. 开始 GRPO 训练 ✓

## 📊 监控训练

**查看实时训练日志**:
```bash
tail -f logs/training.log
```

**查看 GPU 使用情况**:
```bash
watch -n 1 nvidia-smi
```

**查看服务状态**:
```bash
./status.sh
```

## 🛑 停止服务

```bash
./stop.sh
```

或按 `Ctrl+C`（如果在前台运行）

## 📁 关键文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `run.sh` | 一键启动所有服务 |
| `install.sh` | 安装脚本 |
| `stop.sh` | 停止所有服务 |
| `status.sh` | 检查服务状态 |
| `data/` | 训练数据和语料库 |
| `logs/` | 日志文件 |
| `checkpoints/` | 训练检查点 |
| `README.md` | 完整文档 |

## 🔧 常用命令

```bash
# 查看帮助
./run.sh --help

# 仅启动检索服务器
./start_retriever.sh

# 仅运行训练
./train_qwen3_0.6b.sh

# 查看所有日志
ls -lh logs/

# 清理检查点（释放空间）
rm -rf checkpoints/*/checkpoint-*
```

## ⚠️ 常见问题速查

### GPU 内存不足？
➡️ 编辑 `train_qwen3_0.6b.sh`，减小 batch size:
```bash
data.train_batch_size=16  # 改为 16 或 8
```

### 检索服务器无法启动？
➡️ 检查端口占用:
```bash
lsof -i :8000
kill -9 <PID>
```

### 找不到模型？
➡️ 模型会自动从 HuggingFace 下载到:
```
~/.cache/huggingface/hub/
```

确保网络连接正常。

### 训练速度慢？
➡️ 检查 GPU 利用率:
```bash
nvidia-smi
```
如果利用率低，可能是数据加载瓶颈或 batch size 太小。

## 📚 下一步

训练完成后，您可以：

1. **评估模型**: 检查 `checkpoints/` 目录
2. **调整参数**: 编辑 `train_qwen3_0.6b.sh`
3. **使用更大模型**: 改用 Qwen-1.5B 或 Qwen-3B
4. **添加自己的数据**: 编辑 `data/` 目录中的文件
5. **部署模型**: 使用 `infer.py` 进行推理

## 💡 提示

- 首次运行会下载模型（约 1-2GB），需要一些时间
- 示例数据集很小，仅用于演示，实际训练需要更大数据集
- 训练检查点会自动保存在 `checkpoints/` 目录
- 使用 WandB 可以获得更好的训练可视化（需要 API key）

## 🆘 需要帮助？

查看完整文档:
```bash
cat README.md
```

或访问:
- [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1)
- [Search-R1 论文](https://arxiv.org/abs/2503.09516)

---

**开始训练吧！** 🚀✨
