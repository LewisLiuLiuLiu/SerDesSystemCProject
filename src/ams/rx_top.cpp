#include "ams/rx_top.h"
#include <iostream>

namespace serdes {

RxTopModule::RxTopModule(sc_core::sc_module_name nm,
                         const RxParams& rx_params,
                         const AdaptionParams& adaption_params)
    : sc_module(nm)
    , in_p("in_p")
    , in_n("in_n")
    , vdd("vdd")
    , data_out("data_out")
    // TDF signals
    , m_sig_ctle_out_p("sig_ctle_out_p")
    , m_sig_ctle_out_n("sig_ctle_out_n")
    , m_sig_vga_out_p("sig_vga_out_p")
    , m_sig_vga_out_n("sig_vga_out_n")
    , m_sig_dfe_out_p("sig_dfe_out_p")
    , m_sig_dfe_out_n("sig_dfe_out_n")
    , m_sig_sampler_out("sig_sampler_out")
    , m_sig_vref_pos_out("sig_vref_pos_out")
    , m_sig_vref_neg_out("sig_vref_neg_out")
    , m_sig_cdr_phase("sig_cdr_phase")
    , m_sig_cdr_in("sig_cdr_in")
    , m_sig_sampling_trigger("sig_sampling_trigger")
    , m_sig_data_trigger("sig_data_trigger")
    , m_sig_data_feedback("sig_data_feedback")
    , m_sig_clk("sig_clk")
    , m_sig_amp_est_out("sig_amp_est_out")
    // DfeAdaptTdf -> DFE Summer taps
    , m_sig_dfe_tap1_de("sig_dfe_tap1_de")
    , m_sig_dfe_tap2_de("sig_dfe_tap2_de")
    , m_sig_dfe_tap3_de("sig_dfe_tap3_de")
    , m_sig_dfe_tap4_de("sig_dfe_tap4_de")
    , m_sig_dfe_tap5_de("sig_dfe_tap5_de")
    , m_sig_dfe_tap6_de("sig_dfe_tap6_de")
    , m_sig_dfe_tap7_de("sig_dfe_tap7_de")
    , m_sig_dfe_tap8_de("sig_dfe_tap8_de")
    // DfeAdaptTdf -> AdaptionDe statistics
    , m_sig_stat_N_A_de("sig_stat_N_A_de")
    , m_sig_stat_N_B_de("sig_stat_N_B_de")
    , m_sig_stat_N_C_de("sig_stat_N_C_de")
    , m_sig_stat_N_D_de("sig_stat_N_D_de")
    , m_sig_stat_P_pos_de("sig_stat_P_pos_de")
    , m_sig_stat_P_neg_de("sig_stat_P_neg_de")
    , m_sig_stat_C_de("sig_stat_C_de")
    , m_sig_stat_Delta_de("sig_stat_Delta_de")
    , m_sig_stat_state_de("sig_stat_state_de")
    , m_sig_stat_mu_de("sig_stat_mu_de")
    , m_sig_stat_update_count_de("sig_stat_update_count_de")
    , m_sig_vref_current_de("sig_vref_current_de")
    // AdaptionDe -> DfeAdaptTdf control
    , m_sig_mode_de("sig_mode_de")
    , m_sig_reset_de("sig_reset_de")
    , m_sig_vref_cmd_de("sig_vref_cmd_de")
    // AdaptionDe -> VGA
    , m_sig_vga_gain_de("sig_vga_gain_de")
    // AdaptionDe -> Sampler
    , m_sig_sampler_threshold_de("sig_sampler_threshold_de")
    , m_sig_sampler_hysteresis_de("sig_sampler_hysteresis_de")
    , m_sig_zero_vref_de("sig_zero_vref_de")
    // CDR -> AdaptionDe
    , m_sig_phase_error_de("sig_phase_error_de")
    , m_sig_amplitude_rms_de("sig_amplitude_rms_de")
    , m_sig_scenario_switch_de("sig_scenario_switch_de")
    , m_sig_gain_saturated_de("sig_gain_saturated_de")
    // AdaptionDe monitoring outputs
    , m_sig_ctle_zero_de("sig_ctle_zero_de")
    , m_sig_ctle_pole_de("sig_ctle_pole_de")
    , m_sig_ctle_dc_gain_de("sig_ctle_dc_gain_de")
    , m_sig_phase_cmd_de("sig_phase_cmd_de")
    , m_sig_update_count_de("sig_update_count_de")
    , m_sig_freeze_flag_de("sig_freeze_flag_de")
    // Parameters
    , m_params(rx_params)
    , m_adaption_params(adaption_params)
{
    // ========================================================================
    // Instantiate sub-modules (TDF domain)
    // ========================================================================

    m_ctle = new RxCtleTdf("ctle", m_params.ctle);
    m_vga = new RxVgaTdf("vga", m_params.vga);
    m_dfe_summer = new RxDfeSummerTdf("dfe_summer", m_params.dfe_summer);

    // Main sampler: threshold = 0 (data decision d_k)
    RxSamplerParams sampler_params = m_params.sampler;
    sampler_params.phase_source = "phase";
    sampler_params.threshold = 0.0;
    m_sampler = new RxSamplerTdf("sampler", sampler_params);

    // +Vref comparator: threshold = +Vref
    double vref_pos = m_adaption_params.vref_adapt.vref_pos;
    RxSamplerParams vref_pos_params = sampler_params;
    vref_pos_params.threshold = vref_pos;
    vref_pos_params.hysteresis = 0.0;  // No hysteresis for Vref comparators
    m_vref_pos_sampler = new RxSamplerTdf("vref_pos_sampler", vref_pos_params);

    // -Vref comparator: threshold = -Vref
    double vref_neg = m_adaption_params.vref_adapt.vref_neg;
    RxSamplerParams vref_neg_params = sampler_params;
    vref_neg_params.threshold = vref_neg;
    vref_neg_params.hysteresis = 0.0;
    vref_neg_params.threshold_invert = true;  // 动态门限取 -vref_cmd
    m_vref_neg_sampler = new RxSamplerTdf("vref_neg_sampler", vref_neg_params);

    m_cdr = new RxCdrTdf("cdr", m_params.cdr);

    // AGC 幅度检测器：EMA(|v_p - v_n|) at VGA output.
    // tau ≈ 256 UI（默认 ~6.4ns @25ps UI）；DE 刷新为慢路径周期的 1/10，
    // 保证每次慢路径读取时值足够新鲜，同时避免逐采样刷新 DE 事件。
    const double amp_est_tau = 256.0 * m_adaption_params.UI;
    const double vga_sat_level =
        0.5 * (m_params.vga.sat_max - m_params.vga.sat_min);
    const double amp_est_de_interval = m_adaption_params.slow_update_period / 10.0;
    m_amp_est = new AmplitudeEstimatorTdf("amp_est", amp_est_tau,
                                          vga_sat_level, amp_est_de_interval);

    // DFE Adaptation Engine (TDF domain, Plan A)
    m_dfe_adapt = new DfeAdaptTdf("dfe_adapt",
                                   m_adaption_params.dfe,
                                   m_adaption_params.vref_adapt);

    // ========================================================================
    // Instantiate sub-modules (DE domain)
    // ========================================================================

    m_adaption = new AdaptionDe("adaption", m_adaption_params);

    // ========================================================================
    // Instantiate helper modules
    // ========================================================================

    m_clk_src = new ConstClockSource("clk_src");
    m_data_out_latch = new DataStrobeLatchTdf("data_out_latch");
    m_sampler_splitter = new SignalSplitter("sampler_splitter");
    m_dfe_out_splitter = new SignalSplitter3("dfe_out_splitter");

    // ========================================================================
    // Connect TDF signal chain
    // ========================================================================

    // External input -> CTLE -> VGA -> DFE Summer
    m_ctle->in_p(in_p);
    m_ctle->in_n(in_n);
    m_ctle->vdd(vdd);
    m_ctle->out_p(m_sig_ctle_out_p);
    m_ctle->out_n(m_sig_ctle_out_n);

    m_vga->in_p(m_sig_ctle_out_p);
    m_vga->in_n(m_sig_ctle_out_n);
    m_vga->vdd(vdd);
    m_vga->out_p(m_sig_vga_out_p);
    m_vga->out_n(m_sig_vga_out_n);
    // AGC 增益命令（AdaptionDe → VGA）
    m_vga->gain_cmd_de(m_sig_vga_gain_de);
    
    // AGC 幅度检测器：tap VGA 输出；EMA → AdaptionDe；饱和标志 → 互锁
    m_amp_est->in_p(m_sig_vga_out_p);
    m_amp_est->in_n(m_sig_vga_out_n);
    m_amp_est->amplitude(m_sig_amp_est_out);
    m_amp_est->amplitude_de(m_sig_amplitude_rms_de);
    m_amp_est->saturated_de(m_sig_gain_saturated_de);

    m_dfe_summer->in_p(m_sig_vga_out_p);
    m_dfe_summer->in_n(m_sig_vga_out_n);

    // DFE Summer output -> 3-way splitter -> three comparators
    m_dfe_summer->out_p(m_sig_dfe_out_p);
    m_dfe_summer->out_n(m_sig_dfe_out_n);
    m_dfe_out_splitter->in(m_sig_dfe_out_p);
    m_dfe_out_splitter->out1(m_sig_dfe_to_main_in);   // -> Main sampler in_p
    m_dfe_out_splitter->out2(m_sig_dfe_to_vpos_in);   // -> +Vref comparator in_p
    m_dfe_out_splitter->out3(m_sig_dfe_to_vneg_in);   // -> -Vref comparator in_p

    // Connect three comparators (differential: split out_p, use same out_n)
    // Main sampler: in_p from splitter out1, in_n from dfe_out_n
    m_sampler->in_p(m_sig_dfe_to_main_in);
    m_sampler->in_n(m_sig_dfe_out_n);

    // +Vref comparator: same differential input
    m_vref_pos_sampler->in_p(m_sig_dfe_to_vpos_in);
    m_vref_pos_sampler->in_n(m_sig_dfe_out_n);

    // -Vref comparator: same differential input
    m_vref_neg_sampler->in_p(m_sig_dfe_to_vneg_in);
    m_vref_neg_sampler->in_n(m_sig_dfe_out_n);

    // Dummy clock for all samplers
    m_clk_src->out(m_sig_clk);
    m_sampler->clk_sample(m_sig_clk);
    m_vref_pos_sampler->clk_sample(m_sig_clk);
    m_vref_neg_sampler->clk_sample(m_sig_clk);

    // CDR sampling trigger -> main sampler（edge+data 双相 2/UI，供 BBPD）
    m_cdr->sampling_trigger(m_sig_sampling_trigger);
    m_sampler->sampling_trigger(m_sig_sampling_trigger);

    // Vref 比较器由数据频闪驱动（1/UI）：只产生数据中心判决，供 DFE 使用
    m_vref_pos_sampler->sampling_trigger(m_sig_data_trigger);
    m_vref_neg_sampler->sampling_trigger(m_sig_data_trigger);

    // AGC/Vref 闭环（2026-10）：AdaptionDe.vref_cmd → ±Vref 比较器动态门限
    m_vref_pos_sampler->vref_threshold_de(m_sig_vref_cmd_de);
    m_vref_neg_sampler->vref_threshold_de(m_sig_vref_cmd_de);
    // 主采样器保持 slicer@0：门限占位恒 0
    m_sampler->vref_threshold_de(m_sig_zero_vref_de);

    // Main sampler output (data_out) -> downstream
    m_sampler->data_out(m_sig_sampler_out);
    // +Vref comparator output
    m_vref_pos_sampler->data_out(m_sig_vref_pos_out);
    // -Vref comparator output
    m_vref_neg_sampler->data_out(m_sig_vref_neg_out);

    // Main sampler -> splitter -> CDR input + DFE Summer data feedback
    m_sampler_splitter->in(m_sig_sampler_out);
    m_sampler_splitter->out1(m_sig_data_feedback);  // -> DFE Summer
    m_sampler_splitter->out2(m_sig_cdr_in);         // -> CDR

    m_dfe_summer->data_in(m_sig_data_feedback);
    m_dfe_summer->data_strobe(m_sig_data_trigger);   // DFE 历史每 UI 移位一次（P0-3）
    m_cdr->in(m_sig_cdr_in);
    m_cdr->phase_out(m_sig_cdr_phase);
    m_cdr->data_trigger(m_sig_data_trigger);         // 数据频闪（1/UI）
    // BBPD 相位误差 → AdaptionDe（真实接线，2026-10；此前为假常量 0.0）
    m_cdr->phase_error_de(m_sig_phase_error_de);

    // Bind unused data_out_de ports to dummy signals (required by SystemC-AMS)
    m_sampler->data_out_de(m_sig_dummy_data_out_de_0);
    m_vref_pos_sampler->data_out_de(m_sig_dummy_data_out_de_pos);
    m_vref_neg_sampler->data_out_de(m_sig_dummy_data_out_de_neg);

    // External data_out：经 data 频闪锁存，输出干净的 1/UI 数据判决流
    // （主采样器原始流为 edge+data 混合流，含边沿伪比特）
    m_data_out_latch->in(m_sig_sampler_out);
    m_data_out_latch->data_strobe(m_sig_data_trigger);
    m_data_out_latch->out(data_out);

    // ========================================================================
    // Connect DfeAdaptTdf inputs
    // ========================================================================

    // Three comparator outputs -> DfeAdaptTdf
    // Note: data_out is sca_out<double>, carrying 0 or 1
    // We need to split: data_out is already used for CDR/DFE feedback.
    // Use data_out for DfeAdaptTdf's data_in.
    // The +Vref and -Vref comparators have their own data_out ports.

    // Three comparator outputs -> DfeAdaptTdf
    m_dfe_adapt->data_in(m_sig_sampler_out);
    m_dfe_adapt->vref_pos_in(m_sig_vref_pos_out);
    m_dfe_adapt->vref_neg_in(m_sig_vref_neg_out);
    // 自适应频闪 = CDR data_trigger（每 UI 1 个；2026-10 取代内部硬编码 100ps 触发）
    m_dfe_adapt->sampling_trigger(m_sig_data_trigger);

    // DE control inputs
    m_dfe_adapt->mode_de(m_sig_mode_de);
    m_dfe_adapt->reset_de(m_sig_reset_de);
    m_dfe_adapt->vref_cmd_de(m_sig_vref_cmd_de);

    // ========================================================================
    // Connect DfeAdaptTdf outputs -> DFE Summer taps
    // ========================================================================
    m_dfe_adapt->tap1_de(m_sig_dfe_tap1_de);
    m_dfe_adapt->tap2_de(m_sig_dfe_tap2_de);
    m_dfe_adapt->tap3_de(m_sig_dfe_tap3_de);
    m_dfe_adapt->tap4_de(m_sig_dfe_tap4_de);
    m_dfe_adapt->tap5_de(m_sig_dfe_tap5_de);
    m_dfe_adapt->tap6_de(m_sig_dfe_tap6_de);
    m_dfe_adapt->tap7_de(m_sig_dfe_tap7_de);
    m_dfe_adapt->tap8_de(m_sig_dfe_tap8_de);

    m_dfe_summer->tap1_de(m_sig_dfe_tap1_de);
    m_dfe_summer->tap2_de(m_sig_dfe_tap2_de);
    m_dfe_summer->tap3_de(m_sig_dfe_tap3_de);
    m_dfe_summer->tap4_de(m_sig_dfe_tap4_de);
    m_dfe_summer->tap5_de(m_sig_dfe_tap5_de);

    // ========================================================================
    // Connect DfeAdaptTdf statistics -> AdaptionDe
    // ========================================================================
    m_dfe_adapt->stat_N_A(m_sig_stat_N_A_de);
    m_dfe_adapt->stat_N_B(m_sig_stat_N_B_de);
    m_dfe_adapt->stat_N_C(m_sig_stat_N_C_de);
    m_dfe_adapt->stat_N_D(m_sig_stat_N_D_de);
    m_dfe_adapt->stat_P_pos(m_sig_stat_P_pos_de);
    m_dfe_adapt->stat_P_neg(m_sig_stat_P_neg_de);
    m_dfe_adapt->stat_C(m_sig_stat_C_de);
    m_dfe_adapt->stat_Delta(m_sig_stat_Delta_de);
    m_dfe_adapt->stat_state(m_sig_stat_state_de);
    m_dfe_adapt->stat_mu(m_sig_stat_mu_de);
    m_dfe_adapt->stat_update_count(m_sig_stat_update_count_de);
    m_dfe_adapt->vref_current(m_sig_vref_current_de);

    m_adaption->stat_N_A(m_sig_stat_N_A_de);
    m_adaption->stat_N_B(m_sig_stat_N_B_de);
    m_adaption->stat_N_C(m_sig_stat_N_C_de);
    m_adaption->stat_N_D(m_sig_stat_N_D_de);
    m_adaption->stat_P_pos(m_sig_stat_P_pos_de);
    m_adaption->stat_P_neg(m_sig_stat_P_neg_de);
    m_adaption->stat_C(m_sig_stat_C_de);
    m_adaption->stat_Delta(m_sig_stat_Delta_de);
    m_adaption->stat_dfe_state(m_sig_stat_state_de);

    // ========================================================================
    // Connect AdaptionDe -> DfeAdaptTdf (control)
    // ========================================================================
    m_adaption->vref_cmd(m_sig_vref_cmd_de);

    // ========================================================================
    // Connect AdaptionDe other ports
    // ========================================================================

    // AdaptionDe inputs
    m_adaption->phase_error(m_sig_phase_error_de);
    m_adaption->amplitude_rms(m_sig_amplitude_rms_de);
    m_adaption->gain_saturated(m_sig_gain_saturated_de);
    m_adaption->mode(m_sig_mode_de);
    m_adaption->reset(m_sig_reset_de);
    m_adaption->scenario_switch(m_sig_scenario_switch_de);

    // AdaptionDe outputs
    m_adaption->vga_gain(m_sig_vga_gain_de);
    m_adaption->ctle_zero(m_sig_ctle_zero_de);
    m_adaption->ctle_pole(m_sig_ctle_pole_de);
    m_adaption->ctle_dc_gain(m_sig_ctle_dc_gain_de);
    m_adaption->sampler_threshold(m_sig_sampler_threshold_de);
    m_adaption->sampler_hysteresis(m_sig_sampler_hysteresis_de);
    m_adaption->phase_cmd(m_sig_phase_cmd_de);
    m_adaption->update_count(m_sig_update_count_de);
    m_adaption->freeze_flag(m_sig_freeze_flag_de);

    // ========================================================================
    // Initialize DE signals with default values
    // ========================================================================
    // 注意：m_sig_phase_error_de 现由 CDR（sca_de::sca_out）驱动，
    // m_sig_amplitude_rms_de 由幅度检测器驱动，均不能再在此处写入
    // （会触发 sc_signal 多驱动器错误 E115）。
    m_sig_mode_de.write(2);        // Data mode from start
    m_sig_reset_de.write(false);
    m_sig_scenario_switch_de.write(0.0);
}

RxTopModule::~RxTopModule()
{
    delete m_ctle;
    delete m_vga;
    delete m_dfe_summer;
    delete m_sampler;
    delete m_vref_pos_sampler;
    delete m_vref_neg_sampler;
    delete m_cdr;
    delete m_dfe_adapt;
    delete m_amp_est;
    delete m_adaption;
    delete m_clk_src;
    delete m_data_out_latch;
    delete m_sampler_splitter;
    delete m_dfe_out_splitter;
}

double RxTopModule::get_cdr_phase() const
{
    return m_cdr->get_raw_phase();
}

double RxTopModule::get_cdr_integral_state() const
{
    return m_cdr->get_integral_state();
}

double RxTopModule::get_vref_cmd() const
{
    return m_sig_vref_cmd_de.read();
}

} // namespace serdes
