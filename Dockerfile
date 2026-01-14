# ============================================================================
# SerDes SystemC-AMS 开发环境 Dockerfile
# 版本: 1.0
# 描述: 提供完整的 SystemC-2.3.4 + SystemC-AMS-2.3.4 开发环境
# 维护者: Lewisliuliuliu
# ============================================================================

FROM ubuntu:22.04

LABEL maintainer="Lewisliuliuliu"
LABEL description="SerDes SystemC-AMS development environment"
LABEL version="1.0"

# ============================================================================
# 环境变量设置
# ============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV SYSTEMC_VERSION=2.3.4
ENV SYSTEMC_AMS_VERSION=2.3.4
ENV SYSTEMC_HOME=/usr/local/systemc-2.3.4
ENV SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4

# ============================================================================
# 安装基础工具和编译环境
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    clang \
    cmake \
    make \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    libyaml-cpp-dev \
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================================
# 下载并编译安装 SystemC 2.3.4
# ============================================================================
WORKDIR /tmp/build

# 下载 SystemC 源码
RUN wget -q https://github.com/accellera-official/systemc/archive/refs/tags/2.3.4.tar.gz -O systemc-2.3.4.tar.gz \
    && tar -xzf systemc-2.3.4.tar.gz \
    && rm systemc-2.3.4.tar.gz

# 编译安装 SystemC
RUN cd systemc-2.3.4 \
    && mkdir -p build \
    && cd build \
    && cmake .. \
        -DCMAKE_INSTALL_PREFIX=${SYSTEMC_HOME} \
        -DCMAKE_CXX_STANDARD=14 \
        -DBUILD_SHARED_LIBS=OFF \
    && make -j$(nproc) \
    && make install

# 验证 SystemC 安装
RUN test -f ${SYSTEMC_HOME}/lib/libsystemc.a \
    && test -f ${SYSTEMC_HOME}/include/systemc.h \
    && echo "SystemC installation verified successfully"

# ============================================================================
# 下载并编译安装 SystemC-AMS 2.3.4
# 注意: 使用 coseda 维护的 GitHub 镜像
# ============================================================================

# 下载 SystemC-AMS 源码
RUN git clone --depth 1 https://github.com/coseda/systemc-ams.git systemc-ams-2.3.4

# 编译安装 SystemC-AMS (使用 autoconf 构建系统)
RUN cd systemc-ams-2.3.4 \
    && autoreconf -fi \
    && mkdir -p objdir \
    && cd objdir \
    && ../configure \
        --prefix=${SYSTEMC_AMS_HOME} \
        --with-systemc=${SYSTEMC_HOME} \
        CXXFLAGS="-std=c++14" \
    && make -j$(nproc) \
    && make install

# 验证 SystemC-AMS 安装
RUN test -f ${SYSTEMC_AMS_HOME}/lib/libsystemc-ams.a \
    && test -f ${SYSTEMC_AMS_HOME}/include/systemc-ams \
    && echo "SystemC-AMS installation verified successfully"

# ============================================================================
# 清理构建目录
# ============================================================================
RUN rm -rf /tmp/build

# ============================================================================
# 设置库路径
# ============================================================================
ENV LD_LIBRARY_PATH=${SYSTEMC_HOME}/lib:${SYSTEMC_AMS_HOME}/lib:${LD_LIBRARY_PATH}

# ============================================================================
# 创建工作目录
# ============================================================================
WORKDIR /workspace

# ============================================================================
# 健康检查脚本
# ============================================================================
RUN echo '#!/bin/bash\n\
echo "=== SerDes Development Environment ===" \n\
echo "SystemC Home: $SYSTEMC_HOME" \n\
echo "SystemC-AMS Home: $SYSTEMC_AMS_HOME" \n\
echo "" \n\
echo "SystemC Library:" \n\
ls -la $SYSTEMC_HOME/lib/libsystemc.a 2>/dev/null || echo "  [ERROR] Not found" \n\
echo "" \n\
echo "SystemC-AMS Library:" \n\
ls -la $SYSTEMC_AMS_HOME/lib/libsystemc-ams.a 2>/dev/null || echo "  [ERROR] Not found" \n\
echo "" \n\
echo "Build Tools:" \n\
echo "  CMake: $(cmake --version | head -1)" \n\
echo "  Make: $(make --version | head -1)" \n\
echo "  GCC: $(g++ --version | head -1)" \n\
echo "" \n\
echo "Environment ready for SerDes development!" \n\
' > /usr/local/bin/check-env && chmod +x /usr/local/bin/check-env

# ============================================================================
# 默认命令
# ============================================================================
CMD ["/bin/bash"]
