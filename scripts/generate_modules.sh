#!/bin/bash
# 批量生成模块骨架脚本

cd /Users/liuyizhe/.qoder/worktree/serdes/qoder/document-structure-setup-1763903516

# 创建所有AMS模块头文件骨架
cat > include/ams/clock_generation.h << 'EOF'
#ifndef SERDES_CLOCK_GENERATION_H
#define SERDES_CLOCK_GENERATION_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class ClockGenerationTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> clk_phase;
    ClockGenerationTdf(sc_core::sc_module_name nm, const ClockParams& params);
    void set_attributes();
    void processing();
private:
    ClockParams m_params;
    double m_phase;
    double m_frequency;
};
}
#endif
EOF

cat > include/ams/tx_ffe.h << 'EOF'
#ifndef SERDES_TX_FFE_H
#define SERDES_TX_FFE_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class TxFfeTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    TxFfeTdf(sc_core::sc_module_name nm, const TxFfeParams& params);
    void set_attributes();
    void processing();
private:
    TxFfeParams m_params;
    std::vector<double> m_buffer;
};
}
#endif
EOF

cat > include/ams/tx_mux.h << 'EOF'
#ifndef SERDES_TX_MUX_H
#define SERDES_TX_MUX_H
#include <systemc-ams>
namespace serdes {
class TxMuxTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    TxMuxTdf(sc_core::sc_module_name nm);
    void set_attributes();
    void processing();
};
}
#endif
EOF

cat > include/ams/tx_driver.h << 'EOF'
#ifndef SERDES_TX_DRIVER_H
#define SERDES_TX_DRIVER_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class TxDriverTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    TxDriverTdf(sc_core::sc_module_name nm, const TxDriverParams& params);
    void set_attributes();
    void processing();
private:
    TxDriverParams m_params;
};
}
#endif
EOF

cat > include/ams/channel_sparam.h << 'EOF'
#ifndef SERDES_CHANNEL_SPARAM_H
#define SERDES_CHANNEL_SPARAM_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class ChannelSParamTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params);
    void set_attributes();
    void processing();
private:
    ChannelParams m_params;
    std::vector<double> m_buffer;
};
}
#endif
EOF

cat > include/ams/rx_ctle.h << 'EOF'
#ifndef SERDES_RX_CTLE_H
#define SERDES_RX_CTLE_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxCtleTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    RxCtleTdf(sc_core::sc_module_name nm, const RxCtleParams& params);
    void set_attributes();
    void processing();
private:
    RxCtleParams m_params;
};
}
#endif
EOF

cat > include/ams/rx_vga.h << 'EOF'
#ifndef SERDES_RX_VGA_H
#define SERDES_RX_VGA_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxVgaTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    RxVgaTdf(sc_core::sc_module_name nm, const RxVgaParams& params);
    void set_attributes();
    void processing();
private:
    RxVgaParams m_params;
};
}
#endif
EOF

cat > include/ams/rx_sampler.h << 'EOF'
#ifndef SERDES_RX_SAMPLER_H
#define SERDES_RX_SAMPLER_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxSamplerTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_in<double> clk;
    sca_tdf::sca_out<double> out;
    RxSamplerTdf(sc_core::sc_module_name nm, const RxSamplerParams& params);
    void set_attributes();
    void processing();
private:
    RxSamplerParams m_params;
};
}
#endif
EOF

cat > include/ams/rx_dfe.h << 'EOF'
#ifndef SERDES_RX_DFE_H
#define SERDES_RX_DFE_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxDfeTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    RxDfeTdf(sc_core::sc_module_name nm, const RxDfeParams& params);
    void set_attributes();
    void processing();
private:
    RxDfeParams m_params;
    std::vector<double> m_taps;
};
}
#endif
EOF

cat > include/ams/rx_cdr.h << 'EOF'
#ifndef SERDES_RX_CDR_H
#define SERDES_RX_CDR_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxCdrTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> phase_out;
    RxCdrTdf(sc_core::sc_module_name nm, const CdrParams& params);
    void set_attributes();
    void processing();
private:
    CdrParams m_params;
    double m_phase;
};
}
#endif
EOF

echo "All AMS module headers created!"
