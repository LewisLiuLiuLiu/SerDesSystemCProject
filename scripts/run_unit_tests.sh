#!/usr/bin/env bash
# ============================================================================
# 单元测试运行脚本
# 配置 BUILD_TESTING=ON（GoogleTest 缺失时自动获取），构建并串行运行全部测试。
# 单独运行某个测试套件: cd build && ctest --output-on-failure -R <套件名>
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

if [ -z "$SYSTEMC_HOME" ] || [ -z "$SYSTEMC_AMS_HOME" ]; then
    echo "提示: 未设置 SYSTEMC_HOME / SYSTEMC_AMS_HOME，将使用 CMake 自动探测（见 README）"
fi

echo "============================================"
echo "SerDes 单元测试 - 构建并运行"
echo "============================================"

mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DBUILD_TESTING=ON

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cmake --build "$BUILD_DIR" -j"$JOBS"

cd "$BUILD_DIR"
ctest --output-on-failure
