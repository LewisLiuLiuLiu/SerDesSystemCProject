#!/bin/bash
# ============================================================================
# 快速测试脚本 - 运行单个测试套件或测试用例
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
TESTS_DIR="$BUILD_DIR/tests"

# 注：不再硬编码 SystemC 路径——测试可执行文件通过构建期 RPATH 定位库；
# 如需自定义请在调用前 export SYSTEMC_HOME / SYSTEMC_AMS_HOME 并重新构建。

# 使用方法
if [ $# -eq 0 ]; then
    echo "用法: $0 <测试名称>"
    echo ""
    echo "示例:"
    echo "  $0 ctle_basic                    # 运行 ctle_basic 测试"
    echo "  $0 ffe_basic_functionality      # 运行 ffe_basic_functionality 测试"
    echo ""
    echo "注意: 每个测试都是独立的可执行文件，格式为 test_<测试名>"
    echo ""
    echo "可用的测试 (部分示例):"
    echo "  - adaption_* (adaption模块相关测试)"
    echo "  - cdr_* (cdr模块相关测试)"
    echo "  - sampler_* (sampler模块相关测试)"
    echo "  - ffe_* (ffe模块相关测试)"
    echo "  - tx_driver_* (tx_driver模块相关测试)"
    echo "  - wave_gen_* (wave_gen模块相关测试)"
    echo "  - clock_gen_* (clock_gen模块相关测试)"
    echo "  - ctle_basic (ctle模块测试)"
    echo "  - vga_basic (vga模块测试)"
    exit 1
fi

TEST_NAME="$1"
TEST_EXEC="$TESTS_DIR/test_$TEST_NAME"

# 检查测试可执行文件是否存在
if [ ! -f "$TEST_EXEC" ]; then
    echo "错误: 测试可执行文件不存在: $TEST_EXEC"
    echo ""
    echo "可用的测试可执行文件:"
    ls "$TESTS_DIR"/test_* 2>/dev/null | sed 's|.*/test_||' || echo "  (无可用测试)"
    exit 1
fi

echo "============================================"
echo "运行测试: $TEST_NAME"
echo "============================================"
echo ""

cd "$BUILD_DIR"
"$TEST_EXEC"
