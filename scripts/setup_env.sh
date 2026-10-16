#!/bin/bash
# SystemC 环境设置脚本（用 source 运行：source scripts/setup_env.sh）
# 也可复制此文件为 scripts/setup_env_local.sh 后固化你的本地路径
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
    echo "错误: 请用 'source $0' 运行本脚本（直接执行不会影响当前 shell）"
    exit 1
fi

# ============================================================================
# 请根据你的实际安装路径修改以下变量
# ============================================================================

# macOS 示例 (Homebrew 安装)
# export SYSTEMC_HOME=/opt/homebrew/opt/systemc
# export SYSTEMC_AMS_HOME=/opt/homebrew/opt/systemc-ams

# 本地编译安装示例（已设置的环境变量不会被覆盖）
export SYSTEMC_HOME="${SYSTEMC_HOME:-$HOME/systemc-2.3.4}"
export SYSTEMC_AMS_HOME="${SYSTEMC_AMS_HOME:-$HOME/systemc-ams-2.3.4}"

# 或者使用相同目录（如果 SystemC-AMS 安装在 SystemC 目录中）
# export SYSTEMC_AMS_HOME=$SYSTEMC_HOME

# ============================================================================
# 验证路径
# ============================================================================
if [ ! -d "$SYSTEMC_HOME" ]; then
    echo "错误: SYSTEMC_HOME 目录不存在: $SYSTEMC_HOME"
    echo "请修改本文件中的路径为你的实际安装路径"
    return 1
fi

if [ ! -d "$SYSTEMC_AMS_HOME" ]; then
    echo "错误: SYSTEMC_AMS_HOME 目录不存在: $SYSTEMC_AMS_HOME"
    echo "请修改本文件中的路径为你的实际安装路径"
    return 1
fi

# 添加到 PATH（可选）
# export PATH=$SYSTEMC_HOME/bin:$PATH

echo "SystemC 环境已设置:"
echo "  SYSTEMC_HOME:     $SYSTEMC_HOME"
echo "  SYSTEMC_AMS_HOME: $SYSTEMC_AMS_HOME"
