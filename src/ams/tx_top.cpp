#include "ams/tx_top.h"
#include <iostream>

namespace serdes {

TxTopModule::TxTopModule(sc_core::sc_module_name nm, const TxParams& tx_params)
    : sc_module(nm)
    , in("in")
    , vdd("vdd")
    , out_p("out_p")
    , out_n("out_n")
    , m_sig_ffe_out("sig_ffe_out")
    , m_sig_mux_out("sig_mux_out")
    , m_sig_diff_p("sig_diff_p")
    , m_sig_diff_n("sig_diff_n")
    , m_params(tx_params)
{
    // ========================================================================
    // Instantiate sub-modules
    // ========================================================================
    
    // FFE: Feed-Forward Equalizer
    m_ffe = new TxFfeTdf("ffe", m_params.ffe);
    
    // Mux: Lane selection
    m_mux = new TxMuxTdf("mux", m_params.mux_lane);
    
    // S2D: Single-ended to Differential converter
    m_s2d = new SingleToDiffTdf("s2d");
    
    // Driver: Output driver
    m_driver = new TxDriverTdf("driver", m_params.driver);
    
    // ========================================================================
    // Connect signal chain: in → FFE → Mux → S2D → Driver → out
    // ========================================================================
    
    // External input → FFE input
    m_ffe->in(in);
    
    // FFE output → Mux input
    m_ffe->out(m_sig_ffe_out);
    m_mux->in(m_sig_ffe_out);
    
    // Mux output → S2D input
    m_mux->out(m_sig_mux_out);
    m_s2d->in(m_sig_mux_out);
    
    // S2D output → Driver input (differential)
    m_s2d->out_p(m_sig_diff_p);
    m_s2d->out_n(m_sig_diff_n);
    m_driver->in_p(m_sig_diff_p);
    m_driver->in_n(m_sig_diff_n);
    
    // External VDD → Driver VDD
    m_driver->vdd(vdd);
    
    // Driver output → External output (differential)
    m_driver->out_p(out_p);
    m_driver->out_n(out_n);
}

TxTopModule::~TxTopModule()
{
    delete m_ffe;
    delete m_mux;
    delete m_s2d;
    delete m_driver;
}

} // namespace serdes
