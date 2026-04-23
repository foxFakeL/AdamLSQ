#!/bin/bash
# 自动重新安装并测试融合Adam+LSQ优化器

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "融合Adam + LSQ量化优化器 - 自动安装测试脚本"
echo "============================================================"

# 1. 卸载旧包
echo ""
echo "[1/4] 卸载旧版本..."
pip uninstall fused_adam_lsq -y 2>/dev/null || true

# 2. 清除编译缓存
echo ""
echo "[2/4] 清除编译缓存..."
rm -rf /tmp/torch_extensions/fused_adam_lsq* 2>/dev/null || true
rm -rf build/ 2>/dev/null || true
rm -rf dist/ 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
rm -rf src/*.egg-info 2>/dev/null || true

# 3. 安装新包
echo ""
echo "[3/4] 安装新版本..."
pip install -e . --no-build-isolation

# 4. 运行测试
echo ""
echo "[4/4] 运行测试..."
python run_test.py

echo ""
echo "============================================================"
echo "完成!"
echo "============================================================"