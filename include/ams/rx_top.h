#ifndef SERDES_RX_TOP_H
#define SERDES_RX_TOP_H

#include <systemc-ams>
#include "common/parameters.h"
#include "ams/rx_ctle.h"
#include "ams/rx_vga.h"
#include "ams/rx_dfe_summer.h"
#include "ams/rx_sampler.h"
#include "ams/rx_cdr.h"
#include "ams/adaption.h"
#include "ams/dfe_adapt_tdf.h"
#include "ams/amplitude_estimator.h"

namespace serdes {

/**
 * @brief RX Top-level Module
 *
 * Integrates the complete RX signal chain:
 *   Differential Input -> CTLE -> VGA -> DFE Summer -> Samplers (3) <-> CDR
 *                                          |                  |         |
 *                                     DfeAdaptTdf (TDF) ---+         |
 *                                          |                  |
 *                                     AdaptionDe (DE) <-----+
 *
 * Three-comparator architecture for DFE adaptation (per proposal):
 *   - Main sampler: threshold = 0 (data decision d_k)
 *   - +Vref comparator: threshold = +Vref (s_k)
 *   - -Vref comparator: threshold = -Vref (s'_k)
 *
 * DFE adaptation is performed in TDF domain (Plan A) by DfeAdaptTdf.
 * AdaptionDe handles AGC, CDR PI, and Vref command generation in DE domain.
 */
SC_MODULE(RxTopModule) {
public:
    // ========================================================================
    // External Ports
    // ========================================================================
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    sca_tdf::sca_in<double> vdd;
    sca_tdf::sca_out<double> data_out;

    // ========================================================================
    // Constructor & Destructor
    // ========================================================================
    RxTopModule(sc_core::sc_module_name nm,
                const RxParams& rx_params,
                const AdaptionParams& adaption_params);

    ~RxTopModule();

    // ========================================================================
    // Debug Interface
    // ========================================================================
    const sca_tdf::sca_signal<double>& get_ctle_out_p_signal() const {
        return m_sig_ctle_out_p;
    }
    const sca_tdf::sca_signal<double>& get_ctle_out_n_signal() const {
        return m_sig_ctle_out_n;
    }
    const sca_tdf::sca_signal<double>& get_vga_out_p_signal() const {
        return m_sig_vga_out_p;
    }
    const sca_tdf::sca_signal<double>& get_vga_out_n_signal() const {
        return m_sig_vga_out_n;
    }
    const sca_tdf::sca_signal<double>& get_dfe_out_p_signal() const {
        return m_sig_dfe_out_p;
    }
    const sca_tdf::sca_signal<double>& get_dfe_out_n_signal() const {
        return m_sig_dfe_out_n;
    }

    /// AGC amplitude estimator output EMA (monitoring)
    const sca_tdf::sca_signal<double>& get_amplitude_est_signal() const {
        return m_sig_amp_est_out;
    }

    double get_cdr_phase() const;
    double get_cdr_integral_state() const;

    const sca_tdf::sca_signal<double>& get_cdr_phase_signal() const {
        return m_sig_cdr_phase;
    }

    sc_core::sc_signal<double>& get_dfe_tap_signal(int tap_index) {
        switch (tap_index) {
            case 1: return m_sig_dfe_tap1_de;
            case 2: return m_sig_dfe_tap2_de;
            case 3: return m_sig_dfe_tap3_de;
            case 4: return m_sig_dfe_tap4_de;
            case 5: return m_sig_dfe_tap5_de;
            default: return m_sig_dfe_tap1_de;
        }
    }

    /**
     * @brief Get DFE adaptation statistics signal
     */
    sc_core::sc_signal<double>& get_stat_C_signal() {
        return m_sig_stat_C_de;
    }

private:
    // ========================================================================
    // Sub-modules (TDF domain)
    // ========================================================================
    RxCtleTdf* m_ctle;
    RxVgaTdf* m_vga;
    RxDfeSummerTdf* m_dfe_summer;
    RxSamplerTdf* m_sampler;              // Main sampler (thr=0, data decision d_k)
    RxSamplerTdf* m_vref_pos_sampler;     // +Vref comparator (thr=+Vref, s_k)
    RxSamplerTdf* m_vref_neg_sampler;     // -Vref comparator (thr=-Vref, s'_k)
    RxCdrTdf* m_cdr;
    DfeAdaptTdf* m_dfe_adapt;             // DFE adaptation engine (TDF domain)
    AmplitudeEstimatorTdf* m_amp_est;     // AGC amplitude detector (TDF domain)

    // ========================================================================
    // Sub-modules (DE domain)
    // ========================================================================
    AdaptionDe* m_adaption;

    // ========================================================================
    // Internal TDF Signals
    // ========================================================================
    sca_tdf::sca_signal<double> m_sig_ctle_out_p;
    sca_tdf::sca_signal<double> m_sig_ctle_out_n;
    sca_tdf::sca_signal<double> m_sig_vga_out_p;
    sca_tdf::sca_signal<double> m_sig_vga_out_n;
    sca_tdf::sca_signal<double> m_sig_dfe_out_p;
    sca_tdf::sca_signal<double> m_sig_dfe_out_n;
    sca_tdf::sca_signal<double> m_sig_sampler_out;      // Main sampler data_out (output)
    sca_tdf::sca_signal<double> m_sig_vref_pos_out;     // +Vref comparator data_out (output)
    sca_tdf::sca_signal<double> m_sig_vref_neg_out;     // -Vref comparator data_out (output)
    // Splitter outputs to three comparator inputs (input side signals)
    sca_tdf::sca_signal<double> m_sig_dfe_to_main_in;   // DFE out -> main sampler in_p
    sca_tdf::sca_signal<double> m_sig_dfe_to_vpos_in;   // DFE out -> vref_pos sampler in_p
    sca_tdf::sca_signal<double> m_sig_dfe_to_vneg_in;   // DFE out -> vref_neg sampler in_p
    sca_tdf::sca_signal<double> m_sig_cdr_phase;
    sca_tdf::sca_signal<double> m_sig_cdr_in;
    sca_tdf::sca_signal<bool> m_sig_sampling_trigger;
    sca_tdf::sca_signal<double> m_sig_data_feedback;
    sca_tdf::sca_signal<double> m_sig_clk;
    sca_tdf::sca_signal<double> m_sig_amp_est_out;      // AGC 检测器 EMA（监测）

    // ========================================================================
    // DE-TDF Bridge Signals
    // ========================================================================

    // DfeAdaptTdf -> DFE Summer (tap coefficients)
    sc_core::sc_signal<double> m_sig_dfe_tap1_de;
    sc_core::sc_signal<double> m_sig_dfe_tap2_de;
    sc_core::sc_signal<double> m_sig_dfe_tap3_de;
    sc_core::sc_signal<double> m_sig_dfe_tap4_de;
    sc_core::sc_signal<double> m_sig_dfe_tap5_de;
    sc_core::sc_signal<double> m_sig_dfe_tap6_de;
    sc_core::sc_signal<double> m_sig_dfe_tap7_de;
    sc_core::sc_signal<double> m_sig_dfe_tap8_de;

    // DfeAdaptTdf -> AdaptionDe (statistics)
    sc_core::sc_signal<int> m_sig_stat_N_A_de;
    sc_core::sc_signal<int> m_sig_stat_N_B_de;
    sc_core::sc_signal<int> m_sig_stat_N_C_de;
    sc_core::sc_signal<int> m_sig_stat_N_D_de;
    sc_core::sc_signal<double> m_sig_stat_P_pos_de;
    sc_core::sc_signal<double> m_sig_stat_P_neg_de;
    sc_core::sc_signal<double> m_sig_stat_C_de;
    sc_core::sc_signal<double> m_sig_stat_Delta_de;
    sc_core::sc_signal<int> m_sig_stat_state_de;
    sc_core::sc_signal<double> m_sig_stat_mu_de;
    sc_core::sc_signal<int> m_sig_stat_update_count_de;
    sc_core::sc_signal<double> m_sig_vref_current_de;

    // AdaptionDe -> DfeAdaptTdf (control)
    sc_core::sc_signal<int> m_sig_mode_de;
    sc_core::sc_signal<bool> m_sig_reset_de;
    sc_core::sc_signal<double> m_sig_vref_cmd_de;

    // Dummy DE signals for unused data_out_de ports on three comparators
    sc_core::sc_signal<bool> m_sig_dummy_data_out_de_0;
    sc_core::sc_signal<bool> m_sig_dummy_data_out_de_pos;
    sc_core::sc_signal<bool> m_sig_dummy_data_out_de_neg;

    // AdaptionDe -> VGA (gain)
    sc_core::sc_signal<double> m_sig_vga_gain_de;

    // AdaptionDe -> Sampler (threshold/hysteresis, for main sampler only)
    sc_core::sc_signal<double> m_sig_sampler_threshold_de;
    sc_core::sc_signal<double> m_sig_sampler_hysteresis_de;

    // CDR -> AdaptionDe (phase error)
    sc_core::sc_signal<double> m_sig_phase_error_de;
    sc_core::sc_signal<double> m_sig_amplitude_rms_de;   // AGC 检测器 → AdaptionDe
    sc_core::sc_signal<double> m_sig_scenario_switch_de;
    // AGC 饱和互锁（检测器 → AdaptionDe）
    sc_core::sc_signal<bool> m_sig_gain_saturated_de;

    // AdaptionDe outputs (monitoring)
    sc_core::sc_signal<double> m_sig_ctle_zero_de;
    sc_core::sc_signal<double> m_sig_ctle_pole_de;
    sc_core::sc_signal<double> m_sig_ctle_dc_gain_de;
    sc_core::sc_signal<double> m_sig_phase_cmd_de;
    sc_core::sc_signal<int> m_sig_update_count_de;
    sc_core::sc_signal<bool> m_sig_freeze_flag_de;

    // ========================================================================
    // Internal Helper Modules
    // ========================================================================
    class ConstClockSource : public sca_tdf::sca_module {
    public:
        sca_tdf::sca_out<double> out;
        ConstClockSource(sc_core::sc_module_name nm)
            : sca_tdf::sca_module(nm), out("out") {}
        void set_attributes() override { out.set_rate(1); }
        void processing() override { out.write(0.0); }
    };

    class SignalPassThrough : public sca_tdf::sca_module {
    public:
        sca_tdf::sca_in<double> in;
        sca_tdf::sca_out<double> out;
        SignalPassThrough(sc_core::sc_module_name nm)
            : sca_tdf::sca_module(nm), in("in"), out("out") {}
        void set_attributes() override { in.set_rate(1); out.set_rate(1); }
        void processing() override { out.write(in.read()); }
    };

    class SignalSplitter : public sca_tdf::sca_module {
    public:
        sca_tdf::sca_in<double> in;
        sca_tdf::sca_out<double> out1;
        sca_tdf::sca_out<double> out2;
        SignalSplitter(sc_core::sc_module_name nm)
            : sca_tdf::sca_module(nm), in("in"), out1("out1"), out2("out2") {}
        void set_attributes() override { in.set_rate(1); out1.set_rate(1); out2.set_rate(1); }
        void processing() override { double val = in.read(); out1.write(val); out2.write(val); }
    };

    /**
     * @brief Three-way splitter for DFE Summer output to three comparators
     */
    class SignalSplitter3 : public sca_tdf::sca_module {
    public:
        sca_tdf::sca_in<double> in;
        sca_tdf::sca_out<double> out1;
        sca_tdf::sca_out<double> out2;
        sca_tdf::sca_out<double> out3;
        SignalSplitter3(sc_core::sc_module_name nm)
            : sca_tdf::sca_module(nm), in("in"), out1("out1"), out2("out2"), out3("out3") {}
        void set_attributes() override { in.set_rate(1); out1.set_rate(1); out2.set_rate(1); out3.set_rate(1); }
        void processing() override { double val = in.read(); out1.write(val); out2.write(val); out3.write(val); }
    };

    ConstClockSource* m_clk_src;
    SignalPassThrough* m_data_out_tap;
    SignalSplitter* m_sampler_splitter;
    SignalSplitter3* m_dfe_out_splitter;

    // ========================================================================
    // Parameters
    // ========================================================================
    RxParams m_params;
    AdaptionParams m_adaption_params;
};

} // namespace serdes

#endif // SERDES_RX_TOP_H
