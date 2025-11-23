# SerDes SystemC-AMS 构建指南

## 前置要求

- SystemC 2.3.4 已安装
- SystemC-AMS 2.3.4 已安装
- CMake 3.15 或更高版本
- C++14 兼容编译器（Clang 或 GCC）

## 环境设置

首先运行环境设置脚本：

```bash
source scripts/setup_env.sh
```

或者手动设置环境变量：

```bash
export SYSTEMC_HOME=/path/to/systemc-2.3.4
export SYSTEMC_AMS_HOME=/path/to/systemc-ams-2.3.4
```

## 使用 CMake 构建

```bash
# 配置
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# 构建
cmake --build build

# 运行测试
cd build && ctest
```

## 使用 Makefile 构建

```bash
# 查看帮助
make help

# 构建所有目标
make all

# 构建并运行测试
make tests

# 清理
make clean
```

## 构建目标

- `serdes_lib` - 静态库
- `simple_link_tb` - 简单链路测试平台
- `full_system_tb` - 完整系统测试平台
- `unit_tests` - 单元测试
- `integration_tests` - 集成测试

## 故障排除

### 找不到 SystemC 库

确保正确设置 `SYSTEMC_HOME` 和 `SYSTEMC_AMS_HOME` 环境变量。

### 编译错误

确认使用 C++14 标准：
```bash
clang++ --version  # 或 g++ --version
```

最低版本要求：
- GCC 5.0+
- Clang 3.4+
