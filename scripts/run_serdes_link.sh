#!/usr/bin/env bash
# SerDes Link Testbench Runner (NRZ 10 Gbps full link)
# Usage: ./run_serdes_link.sh [scenario] [analyze]
#   scenario: 兼容旧用法保留；nrz_link_tb 当前运行默认 10Gbps 全链路配置
#   analyze:  "yes"/"y" 时运行 Python 后处理（眼图分析 / DFE 抽头绘图）
#
# 输出: build/ 目录下 nrz_10g_*.csv（各节点波形）与 nrz_10g_metadata.json（元数据）

SCENARIO=${1:-basic}
ANALYZE=${2:-no}

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "=========================================="
echo "SerDes Link Testbench (nrz_link_tb)"
echo "=========================================="
echo "（场景参数: $SCENARIO，暂不使用；当前运行默认 10Gbps 全链路）"
echo ""

# Build if needed
echo "Building testbench..."
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" > /dev/null
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cmake --build "$BUILD_DIR" --target nrz_link_tb -j"$JOBS"

# Run simulation (writes nrz_10g_*.csv into the build directory)
echo ""
echo "Running simulation..."
cd "$BUILD_DIR"
./bin/nrz_link_tb

# Analyze with EyeAnalyzer and DFE tap plotter if requested
if [ "$ANALYZE" = "yes" ] || [ "$ANALYZE" = "y" ]; then
    echo ""
    echo "Running EyeAnalyzer..."
    cd "$PROJECT_ROOT"
    python3 scripts/analyze_serdes_link.py nrz_10g

    echo ""
    echo "Plotting DFE tap coefficients..."
    python3 scripts/plot_dfe_taps.py build/nrz_10g_dfe_taps.csv
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
