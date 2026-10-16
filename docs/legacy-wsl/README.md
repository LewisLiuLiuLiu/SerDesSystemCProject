# Legacy：WSL-only 构建脚本（保留存档）

这两个脚本来自早期 Windows WSL 开发环境，**不可跨平台运行**（硬编码 `/mnt/d` 路径、`readelf/ldd`、`LD_LIBRARY_PATH` 等），仅作历史参考与故障回溯，请勿在 macOS / 常规 Linux 环境中执行。

- `fix_systemc_ams.sh` — 修复 SystemC-AMS 安装（WSL 环境）
- `rebuild_systemc_ams.sh` — 重建 SystemC-AMS（WSL 环境）

正确的构建方式见仓库根目录 `README.md`（CMake + `-DSYSTEMC_HOME` / `-DSYSTEMC_AMS_HOME`，或环境变量）。
