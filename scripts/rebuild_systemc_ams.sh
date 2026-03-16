#!/bin/bash
# Rebuild SystemC-AMS with correct SystemC 2.3.4 path

set -e

SYSTEMC_ROOT=/mnt/d/systemCProjects/systemCsrc/systemc-2.3.4-install
SYSTEMCAMS_SRC=/mnt/d/systemCProjects/systemCsrc/systemc-ams-2.3.4-src
SYSTEMCAMS_INSTALL=/mnt/d/systemCProjects/systemCsrc/systemc-ams-install

echo "=== Cleaning old SystemC-AMS installation ==="
rm -rf $SYSTEMCAMS_INSTALL
mkdir -p $SYSTEMCAMS_INSTALL

echo "=== Configuring SystemC-AMS ==="
cd $SYSTEMCAMS_SRC
rm -rf build
mkdir build && cd build

# Explicitly set SystemC paths
export SYSTEMC_HOME=$SYSTEMC_ROOT
export LD_LIBRARY_PATH=$SYSTEMC_ROOT/lib:$LD_LIBRARY_PATH

cmake .. \
    -DCMAKE_INSTALL_PREFIX=$SYSTEMCAMS_INSTALL \
    -DCMAKE_PREFIX_PATH=$SYSTEMC_ROOT \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_CXX_FLAGS="-I$SYSTEMC_ROOT/include" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$SYSTEMC_ROOT/lib -lsystemc"

echo "=== Building SystemC-AMS ==="
make -j4

echo "=== Installing SystemC-AMS ==="
make install

echo "=== Verifying library dependencies ==="
echo "Checking libsystemc-ams.so dependencies:"
ldd $SYSTEMCAMS_INSTALL/liblinux64/libsystemc-ams.so | grep systemc

echo "=== Done ==="
