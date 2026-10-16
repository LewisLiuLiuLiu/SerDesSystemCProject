/**
 * @file adaption_test_common.h
 * @brief Common test infrastructure for AdaptionDe unit tests
 *
 * NOTE (2026-10): DFE tap updates moved to DfeAdaptTdf, and sampler
 * threshold/hysteresis are no longer adapted by AdaptionDe (fixed defaults).
 * This fixture drives the current interface: AGC, CDR PI, Vref command
 * generation and safety mechanisms, with statistics inputs provided by a
 * StatsSource (stand-in for DfeAdaptTdf outputs).
 */

#ifndef SERDES_TESTS_ADAPTION_TEST_COMMON_H
#define SERDES_TESTS_ADAPTION_TEST_COMMON_H

#include <gtest/gtest.h>
#include <systemc>
#include <cmath>
#include "ams/adaption.h"
#include "common/parameters.h"

namespace serdes {
namespace test {

// ============================================================================
// Simple signal source for unit testing
// ============================================================================
class SimpleAdaptionSource : public sc_core::sc_module {
public:
    sc_core::sc_out<double> phase_error;
    sc_core::sc_out<double> amplitude_rms;
    sc_core::sc_out<int> error_count;
    sc_core::sc_out<double> isi_metric;
    sc_core::sc_out<int> mode;
    sc_core::sc_out<bool> reset;
    sc_core::sc_out<double> scenario_switch;
    
    double m_phase_error;
    double m_amplitude;
    int m_error_count;
    int m_mode;
    
    SC_HAS_PROCESS(SimpleAdaptionSource);
    
    SimpleAdaptionSource(sc_core::sc_module_name nm)
        : sc_core::sc_module(nm)
        , phase_error("phase_error")
        , amplitude_rms("amplitude_rms")
        , error_count("error_count")
        , isi_metric("isi_metric")
        , mode("mode")
        , reset("reset")
        , scenario_switch("scenario_switch")
        , m_phase_error(0.5e-11)
        , m_amplitude(0.3)
        , m_error_count(0)
        , m_mode(2)
    {
        SC_THREAD(source_process);
    }
    
    void source_process() {
        // 先写入正常初值：避免 reset 窗口内信号保持默认 0 触发安全机制误判
        phase_error.write(m_phase_error);
        amplitude_rms.write(m_amplitude);
        error_count.write(m_error_count);
        isi_metric.write(0.1);
        mode.write(m_mode);
        scenario_switch.write(0.0);
        
        // Initial reset
        reset.write(true);
        wait(sc_core::sc_time(10, sc_core::SC_NS));
        reset.write(false);
        
        while (true) {
            phase_error.write(m_phase_error);
            amplitude_rms.write(m_amplitude);
            error_count.write(m_error_count);
            isi_metric.write(0.1);
            mode.write(m_mode);
            scenario_switch.write(0.0);
            
            wait(sc_core::sc_time(1, sc_core::SC_NS));
        }
    }
    
    void set_phase_error(double val) { m_phase_error = val; }
    void set_amplitude(double val) { m_amplitude = val; }
    void set_error_count(int val) { m_error_count = val; }
    void set_mode(int val) { m_mode = val; }
};

// ============================================================================
// Statistics source (stand-in for DfeAdaptTdf statistics outputs)
// ============================================================================
class StatsSource : public sc_core::sc_module {
public:
    sc_core::sc_out<int> N_A;
    sc_core::sc_out<int> N_B;
    sc_core::sc_out<int> N_C;
    sc_core::sc_out<int> N_D;
    sc_core::sc_out<double> P_pos;
    sc_core::sc_out<double> P_neg;
    sc_core::sc_out<double> C;
    sc_core::sc_out<double> Delta;
    sc_core::sc_out<int> dfe_state;
    
    int m_N_A;
    int m_N_B;
    int m_N_C;
    int m_N_D;
    double m_P_pos;
    double m_P_neg;
    double m_C;
    double m_Delta;
    int m_dfe_state;
    
    SC_HAS_PROCESS(StatsSource);
    
    StatsSource(sc_core::sc_module_name nm)
        : sc_core::sc_module(nm)
        , N_A("N_A")
        , N_B("N_B")
        , N_C("N_C")
        , N_D("N_D")
        , P_pos("P_pos")
        , P_neg("P_neg")
        , C("C")
        , Delta("Delta")
        , dfe_state("dfe_state")
        , m_N_A(0)
        , m_N_B(0)
        , m_N_C(0)
        , m_N_D(0)
        , m_P_pos(0.0)
        , m_P_neg(0.0)
        , m_C(0.0)
        , m_Delta(0.0)
        , m_dfe_state(0)
    {
        SC_THREAD(source_process);
    }
    
    void source_process() {
        while (true) {
            N_A.write(m_N_A);
            N_B.write(m_N_B);
            N_C.write(m_N_C);
            N_D.write(m_N_D);
            P_pos.write(m_P_pos);
            P_neg.write(m_P_neg);
            C.write(m_C);
            Delta.write(m_Delta);
            dfe_state.write(m_dfe_state);
            
            wait(sc_core::sc_time(10, sc_core::SC_NS));
        }
    }
    
    void set_confidence(double val) { m_C = val; }
    void set_delta(double val) { m_Delta = val; }
    void set_dfe_state(int val) { m_dfe_state = val; }
};

// ============================================================================
// Test fixture for Adaption unit tests
// ============================================================================
class AdaptionBasicTestbench : public sc_core::sc_module {
public:
    SimpleAdaptionSource* src;
    StatsSource* stats;
    AdaptionDe* adaption;
    
    // Signals
    sc_core::sc_signal<double> sig_phase_error;
    sc_core::sc_signal<double> sig_amplitude_rms;
    sc_core::sc_signal<int> sig_error_count;
    sc_core::sc_signal<double> sig_isi_metric;
    sc_core::sc_signal<int> sig_mode;
    sc_core::sc_signal<bool> sig_reset;
    sc_core::sc_signal<double> sig_scenario_switch;
    sc_core::sc_signal<bool> sig_gain_saturated;  // AGC 饱和互锁输入
    
    // Statistics signals (from StatsSource)
    sc_core::sc_signal<int> sig_stat_N_A;
    sc_core::sc_signal<int> sig_stat_N_B;
    sc_core::sc_signal<int> sig_stat_N_C;
    sc_core::sc_signal<int> sig_stat_N_D;
    sc_core::sc_signal<double> sig_stat_P_pos;
    sc_core::sc_signal<double> sig_stat_P_neg;
    sc_core::sc_signal<double> sig_stat_C;
    sc_core::sc_signal<double> sig_stat_Delta;
    sc_core::sc_signal<int> sig_stat_dfe_state;
    
    sc_core::sc_signal<double> sig_vga_gain;
    sc_core::sc_signal<double> sig_ctle_zero;
    sc_core::sc_signal<double> sig_ctle_pole;
    sc_core::sc_signal<double> sig_ctle_dc_gain;
    sc_core::sc_signal<double> sig_vref_cmd;
    sc_core::sc_signal<double> sig_sampler_threshold;
    sc_core::sc_signal<double> sig_sampler_hysteresis;
    sc_core::sc_signal<double> sig_phase_cmd;
    sc_core::sc_signal<int> sig_update_count;
    sc_core::sc_signal<bool> sig_freeze_flag;
    
    AdaptionParams params;
    
    AdaptionBasicTestbench(sc_core::sc_module_name nm, const AdaptionParams& p)
        : sc_core::sc_module(nm)
        , params(p)
    {
        src = new SimpleAdaptionSource("src");
        stats = new StatsSource("stats");
        adaption = new AdaptionDe("adaption", params);
        
        // Connect source
        src->phase_error(sig_phase_error);
        src->amplitude_rms(sig_amplitude_rms);
        src->error_count(sig_error_count);   // 保留（当前未接入 AdaptionDe）
        src->isi_metric(sig_isi_metric);     // 保留（当前未接入 AdaptionDe）
        src->mode(sig_mode);
        src->reset(sig_reset);
        src->scenario_switch(sig_scenario_switch);
        
        // Connect statistics source
        stats->N_A(sig_stat_N_A);
        stats->N_B(sig_stat_N_B);
        stats->N_C(sig_stat_N_C);
        stats->N_D(sig_stat_N_D);
        stats->P_pos(sig_stat_P_pos);
        stats->P_neg(sig_stat_P_neg);
        stats->C(sig_stat_C);
        stats->Delta(sig_stat_Delta);
        stats->dfe_state(sig_stat_dfe_state);
        
        // Connect adaption inputs
        adaption->phase_error(sig_phase_error);
        adaption->amplitude_rms(sig_amplitude_rms);
        adaption->gain_saturated(sig_gain_saturated);
        adaption->mode(sig_mode);
        adaption->reset(sig_reset);
        adaption->scenario_switch(sig_scenario_switch);
        adaption->stat_N_A(sig_stat_N_A);
        adaption->stat_N_B(sig_stat_N_B);
        adaption->stat_N_C(sig_stat_N_C);
        adaption->stat_N_D(sig_stat_N_D);
        adaption->stat_P_pos(sig_stat_P_pos);
        adaption->stat_P_neg(sig_stat_P_neg);
        adaption->stat_C(sig_stat_C);
        adaption->stat_Delta(sig_stat_Delta);
        adaption->stat_dfe_state(sig_stat_dfe_state);
        
        // Connect adaption outputs
        adaption->vga_gain(sig_vga_gain);
        adaption->ctle_zero(sig_ctle_zero);
        adaption->ctle_pole(sig_ctle_pole);
        adaption->ctle_dc_gain(sig_ctle_dc_gain);
        adaption->vref_cmd(sig_vref_cmd);
        adaption->sampler_threshold(sig_sampler_threshold);
        adaption->sampler_hysteresis(sig_sampler_hysteresis);
        adaption->phase_cmd(sig_phase_cmd);
        adaption->update_count(sig_update_count);
        adaption->freeze_flag(sig_freeze_flag);
    }
    
    ~AdaptionBasicTestbench() {
        // SystemC modules are automatically managed by the simulator
        // Do not delete them manually
    }
    
    double get_vga_gain() { return sig_vga_gain.read(); }
    double get_phase_cmd() { return sig_phase_cmd.read(); }
    void set_gain_saturated(bool v) { sig_gain_saturated.write(v); }
    double get_threshold() { return sig_sampler_threshold.read(); }
    double get_hysteresis() { return sig_sampler_hysteresis.read(); }
    double get_vref_cmd() { return sig_vref_cmd.read(); }
    int get_update_count() { return sig_update_count.read(); }
    bool get_freeze_flag() { return sig_freeze_flag.read(); }
};

} // namespace test
} // namespace serdes

#endif // SERDES_TESTS_ADAPTION_TEST_COMMON_H
