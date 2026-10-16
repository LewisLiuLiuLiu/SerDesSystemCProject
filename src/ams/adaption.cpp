/**
 * @file adaption.cpp
 * @brief AdaptionDe class implementation - DE domain adaptive control module
 *
 * Implements DE-domain adaptive control algorithms:
 * - AGC (Automatic Gain Control) with PI controller
 * - CDR PI controller
 * - Vref command generation (for DfeAdaptTdf)
 * - Safety mechanisms: freeze on error, rollback to snapshot
 *
 * Note: DFE tap update has been moved to DfeAdaptTdf (TDF domain).
 *       AdaptionDe reads statistics from DfeAdaptTdf and generates Vref commands.
 */

#include "ams/adaption.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace serdes {

// ============================================================================
// Constructor
// ============================================================================
AdaptionDe::AdaptionDe(sc_core::sc_module_name nm, const AdaptionParams& params)
    : sc_core::sc_module(nm)
    // Input ports
    , phase_error("phase_error")
    , amplitude_rms("amplitude_rms")
    , gain_saturated("gain_saturated")
    , mode("mode")
    , reset("reset")
    , scenario_switch("scenario_switch")
    // Statistics inputs from DfeAdaptTdf
    , stat_N_A("stat_N_A")
    , stat_N_B("stat_N_B")
    , stat_N_C("stat_N_C")
    , stat_N_D("stat_N_D")
    , stat_P_pos("stat_P_pos")
    , stat_P_neg("stat_P_neg")
    , stat_C("stat_C")
    , stat_Delta("stat_Delta")
    , stat_dfe_state("stat_dfe_state")
    // Output ports
    , vga_gain("vga_gain")
    , ctle_zero("ctle_zero")
    , ctle_pole("ctle_pole")
    , ctle_dc_gain("ctle_dc_gain")
    , vref_cmd("vref_cmd")
    , sampler_threshold("sampler_threshold")
    , sampler_hysteresis("sampler_hysteresis")
    , phase_cmd("phase_cmd")
    , update_count("update_count")
    , freeze_flag("freeze_flag")
    // Parameters
    , m_params(params)
{
    // 参数校验（fail-fast）：非法周期会让 wait() 行为异常（0 → 死循环，
    // 负值/NaN → 时间溢出），必须在构造期拒绝。
    if (!(params.fast_update_period > 0.0) || !(params.slow_update_period > 0.0)) {
        throw std::invalid_argument(
            "AdaptionDe: fast_update_period and slow_update_period must be > 0");
    }

    // Initialize timing
    m_fast_period = sc_core::sc_time(params.fast_update_period, sc_core::SC_SEC);
    m_slow_period = sc_core::sc_time(params.slow_update_period, sc_core::SC_SEC);

    // Initialize state
    initialize_state();

    // Register processes
    SC_THREAD(fast_path_process);
    sensitive << reset;

    SC_THREAD(slow_path_process);
    sensitive << reset;

    SC_METHOD(reset_process);
    sensitive << reset.pos();
    dont_initialize();
}

// ============================================================================
// Destructor
// ============================================================================
AdaptionDe::~AdaptionDe() {
}

// ============================================================================
// State Initialization
// ============================================================================
void AdaptionDe::initialize_state() {
    // AGC state
    m_agc_integral = 0.0;
    m_current_gain = m_params.agc.initial_gain;
    m_prev_gain = m_current_gain;

    // CDR PI state
    m_cdr_integral = 0.0;
    m_current_phase_cmd = m_params.cdr_pi.initial_phase;

    // Threshold state (fixed defaults, no longer adapted)
    m_current_threshold = 0.0;
    m_current_hysteresis = 0.02;

    // Vref command state
    m_current_vref_cmd = m_params.vref_adapt.vref_pos;

    // Update tracking
    m_update_count = 0;
    m_fast_update_count = 0;
    m_slow_update_count = 0;
    m_freeze_flag = false;
    m_signal_present = false;

    // Snapshot management
    m_snapshots.clear();
    m_snapshots.reserve(100);
    m_last_snapshot_time = sc_core::SC_ZERO_TIME;
}

// ============================================================================
// Reset Handler
// ============================================================================
void AdaptionDe::reset_process() {
    if (reset.read()) {
        initialize_state();
    }
}

// ============================================================================
// Fast Path Process (CDR PI)
// ============================================================================
void AdaptionDe::fast_path_process() {
    wait(sc_core::SC_ZERO_TIME);

    // Initialize fast path outputs
    sampler_threshold.write(m_current_threshold);
    sampler_hysteresis.write(m_current_hysteresis);
    phase_cmd.write(m_current_phase_cmd);
    update_count.write(m_update_count);
    freeze_flag.write(m_freeze_flag);

    while (true) {
        wait(m_fast_period);

        update_signal_present();

        int current_mode = mode.read();
        if (current_mode == 3 || m_freeze_flag) {
            continue;
        }

        // Check for freeze conditions
        if (m_params.safety.freeze_on_error && check_freeze_condition()) {
            m_freeze_flag = true;
            freeze_flag.write(true);

            if (m_params.safety.rollback_enable) {
                rollback_to_snapshot();
            }
            continue;
        }

        // CDR PI Update
        if (m_params.cdr_pi.enabled && (current_mode == 1 || current_mode == 2)) {
            double phase_err = phase_error.read();
            m_current_phase_cmd = cdr_pi_update(phase_err);
            phase_cmd.write(m_current_phase_cmd);
        }

        // Update counters
        m_fast_update_count++;
        m_update_count++;
        update_count.write(m_update_count);
    }
}

// ============================================================================
// Slow Path Process (AGC + Vref command)
// ============================================================================
void AdaptionDe::slow_path_process() {
    wait(sc_core::SC_ZERO_TIME);

    // Write initial outputs
    write_all_outputs();

    while (true) {
        wait(m_slow_period);

        update_signal_present();

        int current_mode = mode.read();
        if (current_mode == 3 || m_freeze_flag) {
            continue;
        }

        // Snapshot Management
        sc_core::sc_time current_time = sc_core::sc_time_stamp();
        sc_core::sc_time snapshot_interval(m_params.safety.snapshot_interval, sc_core::SC_SEC);

        if ((current_time - m_last_snapshot_time) >= snapshot_interval) {
            save_snapshot();
            m_last_snapshot_time = current_time;
        }

        // AGC Update
        // 信号尚未建立前不更新（避免空输入把增益盲推到上限；
        // 检测器 EMA 建立后（> 存活阈值）恢复正常 PI 控制）。
        if (m_params.agc.enabled && m_signal_present &&
            (current_mode == 1 || current_mode == 2)) {
            double amp = amplitude_rms.read();
            m_current_gain = agc_pi_update(amp);
            vga_gain.write(m_current_gain);
        }

        // Vref command generation
        if (m_params.dfe.enabled && m_params.vref_adapt.enabled) {
            update_vref_command();
            vref_cmd.write(m_current_vref_cmd);
        }

        // Update counters
        m_slow_update_count++;
    }
}

// ============================================================================
// AGC PI Controller Update（位置式 PI，2026-10 修复）
// ============================================================================
// 旧实现 new_gain = current_gain + P + I 把积分项累积到“当前增益”上，
// 等效于对积分做了二次积分（增益以 t² 发散）。正确的位置式形式为：
//   gain = gain_initial + Kp * e + Σ Ki * e * T
// 另加入 (v1) 饱和互锁：VGA 输出接近限幅时误差信号失真，禁止上调用，
// 并做积分限幅防止风卷。
double AdaptionDe::agc_pi_update(double amplitude) {
    double amp_error = m_params.agc.target_amplitude - amplitude;

    // 饱和互锁（v1）：接近限幅时增益上调已无法提升幅度 → 冻结正误差
    if (gain_saturated.read() && amp_error > 0.0) {
        amp_error = 0.0;
    }

    double P = m_params.agc.kp * amp_error;
    m_agc_integral += m_params.agc.ki * amp_error * m_params.slow_update_period;

    // 积分限幅：保证 initial + I 不逃逸 [gain_min, gain_max] 量程
    m_agc_integral = clamp(m_agc_integral,
                           m_params.agc.gain_min - m_params.agc.initial_gain,
                           m_params.agc.gain_max - m_params.agc.initial_gain);

    // 位置式：以配置初值 + (P + I) 给出绝对增益
    double new_gain = m_params.agc.initial_gain + P + m_agc_integral;

    new_gain = clamp(new_gain, m_params.agc.gain_min, m_params.agc.gain_max);

    double max_change = m_params.agc.rate_limit * m_params.slow_update_period;
    double gain_change = new_gain - m_prev_gain;
    if (std::abs(gain_change) > max_change) {
        double sign = (gain_change > 0) ? 1.0 : -1.0;
        new_gain = m_prev_gain + sign * max_change;
    }

    new_gain = clamp(new_gain, m_params.agc.gain_min, m_params.agc.gain_max);

    m_prev_gain = new_gain;
    return new_gain;
}

// ============================================================================
// CDR PI Controller Update
// ============================================================================
double AdaptionDe::cdr_pi_update(double phase_err) {
    double P = m_params.cdr_pi.kp * phase_err;

    double new_integral = m_cdr_integral + m_params.cdr_pi.ki * phase_err * m_params.fast_update_period;
    double new_phase_cmd = P + new_integral;

    double phase_range = m_params.cdr_pi.phase_range;
    if (std::abs(new_phase_cmd) > phase_range) {
        new_phase_cmd = clamp(new_phase_cmd, -phase_range, phase_range);

        if (!m_params.cdr_pi.anti_windup) {
            m_cdr_integral = new_integral;
        }
    } else {
        m_cdr_integral = new_integral;
    }

    double resolution = m_params.cdr_pi.phase_resolution;
    new_phase_cmd = std::round(new_phase_cmd / resolution) * resolution;

    return new_phase_cmd;
}

// ============================================================================
// Vref Command Generation
// ============================================================================
void AdaptionDe::update_vref_command() {
    // Read statistics from DfeAdaptTdf
    double C = stat_C.read();
    double Delta = stat_Delta.read();
    int dfe_state = stat_dfe_state.read();

    // Strategy depends on DFE state machine state
    switch (dfe_state) {
    case 0:  // STARTUP
        // Use initial Vref, let DfeAdaptTdf self-adapt
        m_current_vref_cmd = m_params.vref_adapt.vref_initial;
        break;

    case 1:  // ACQUIRE
        // Fine-tune Vref based on confidence
        // Target: C = target_confidence
        {
            double error = C - m_params.vref_adapt.target_confidence;
            m_current_vref_cmd *= (1.0 + m_params.vref_adapt.adapt_alpha * error);
            m_current_vref_cmd = clamp(m_current_vref_cmd, 0.01, 0.5);
        }
        break;

    case 2:  // TRACK
        // Slow tracking, only adjust if Delta is large
        if (std::abs(Delta) > m_params.vref_adapt.asymmetry_alert) {
            // Asymmetry too large, adjust Vref to compensate
            double correction = m_params.vref_adapt.adapt_alpha * 0.1 * Delta;
            m_current_vref_cmd -= correction;
            m_current_vref_cmd = clamp(m_current_vref_cmd, 0.01, 0.5);
        }
        break;

    default:
        m_current_vref_cmd = m_params.vref_adapt.vref_pos;
        break;
    }
}

// ============================================================================
// Safety Methods
// ============================================================================
bool AdaptionDe::check_freeze_condition() {
    // 幅度异常：信号已建立后消失（< 0.01）才算异常；
    // 仿真启动阶段（信号尚未建立）不触发幅度冻结。
    double amp = amplitude_rms.read();
    if (m_signal_present && amp < 0.01) {
        return true;
    }
    if (amp > 2.0) {
        return true;
    }

    // Check for phase error out of range
    double phase_err = phase_error.read();
    if (std::abs(phase_err) > m_params.cdr_pi.phase_range * 2.0) {
        return true;
    }

    return false;
}

void AdaptionDe::update_signal_present() {
    // 存活阈值 0.02V：大于它认为信号已建立（对 EMA 检测器而言
    // 启动斜坡需要数个 tau，短仿真窗口内不会误置位）。
    if (!m_signal_present && amplitude_rms.read() > 0.02) {
        m_signal_present = true;
    }
}

void AdaptionDe::save_snapshot() {
    Snapshot snap;
    snap.vga_gain = m_current_gain;
    snap.threshold = m_current_threshold;
    snap.hysteresis = m_current_hysteresis;
    snap.phase_cmd = m_current_phase_cmd;
    snap.vref_cmd = m_current_vref_cmd;
    snap.timestamp = sc_core::sc_time_stamp();
    snap.valid = true;

    if (m_snapshots.size() >= 100) {
        m_snapshots.erase(m_snapshots.begin());
    }
    m_snapshots.push_back(snap);
}

bool AdaptionDe::rollback_to_snapshot() {
    for (int i = static_cast<int>(m_snapshots.size()) - 1; i >= 0; --i) {
        if (m_snapshots[i].valid) {
            const Snapshot& snap = m_snapshots[i];

            m_current_gain = snap.vga_gain;
            m_current_threshold = snap.threshold;
            m_current_hysteresis = snap.hysteresis;
            m_current_phase_cmd = snap.phase_cmd;
            m_current_vref_cmd = snap.vref_cmd;

            m_agc_integral = 0.0;
            m_cdr_integral = 0.0;

            // 注意：此处不能调用 write_all_outputs()——这些输出端口已由 slow_path
            // 驱动，快路径再写会触发 sc_signal 多驱动器错误 (E115)。回滚仅恢复内部
            // 状态；冻结点后的端口输出保持冻结前值，解冻后由慢路径重新发布。

            m_freeze_flag = false;
            return true;
        }
    }
    return false;
}

// ============================================================================
// Helper Methods
// ============================================================================
double AdaptionDe::clamp(double val, double min_val, double max_val) const {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

void AdaptionDe::write_all_outputs() {
    vga_gain.write(m_current_gain);

    ctle_zero.write(2e9);
    ctle_pole.write(30e9);
    ctle_dc_gain.write(1.5);

    vref_cmd.write(m_current_vref_cmd);
}

} // namespace serdes
