/**
 * @file rx_cdr.cpp
 * @brief Implementation of Clock and Data Recovery (CDR) module
 * 
 * @version 0.2
 * @date 2026-01-20
 */

#include "ams/rx_cdr.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace serdes {

// ============================================================================
// Constructor
// ============================================================================

RxCdrTdf::RxCdrTdf(sc_core::sc_module_name nm, const CdrParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , phase_out("phase_out")
    , sampling_trigger("sampling_trigger")
    , phase_error_de("phase_error_de")
    , m_params(params)
    , m_sample_state(SampleState::WAIT_EDGE)
    , m_edge_sample(false)
    , m_data_sample(false)
    , m_prev_data_sample(false)
    , m_phase(0.0)
    , m_integral(0.0)
    , m_last_phase_error(0.0)
    , m_step_index(0)
    , m_edge_ui(0)
    , m_data_ui(0)
    , m_next_edge_step(0)
    , m_next_data_step(0)
    , m_scheduled_qq(0.0)
    , m_ts_cached(0.0)
    , m_last_written_pe(1.0)  // 哨兵：与任何合法误差值（0/±ui）不同，强制首次写出
{
    // Validate parameters during construction
    validate_params();
}

// ============================================================================
// Parameter Validation
// ============================================================================

void RxCdrTdf::validate_params()
{
    // Validate PI controller parameters
    if (m_params.pi.kp < 0.0) {
        throw std::invalid_argument("CDR: Kp must be non-negative");
    }
    if (m_params.pi.ki < 0.0) {
        throw std::invalid_argument("CDR: Ki must be non-negative");
    }
    if (m_params.pi.edge_threshold <= 0.0) {
        throw std::invalid_argument("CDR: edge_threshold must be positive");
    }
    
    // Validate phase interpolator parameters
    if (m_params.pai.resolution <= 0.0) {
        throw std::invalid_argument("CDR: PAI resolution must be positive");
    }
    if (m_params.pai.range <= 0.0) {
        throw std::invalid_argument("CDR: PAI range must be positive");
    }
    if (m_params.pai.range < m_params.pai.resolution) {
        throw std::invalid_argument("CDR: PAI range must be >= resolution");
    }
}

// ============================================================================
// TDF Attribute Setup
// ============================================================================

void RxCdrTdf::set_attributes()
{
    // Set port rates (1:1 processing)
    in.set_rate(1);
    phase_out.set_rate(1);
    phase_out.set_delay(1);  // 添加延迟以打破 Sampler-CDR 反馈环路
    sampling_trigger.set_rate(1);
    sampling_trigger.set_delay(1);  // 添加延迟以打破环路
}

// ============================================================================
// Initialization
// ============================================================================

void RxCdrTdf::initialize()
{
    // Reset internal state
    m_phase = 0.0;
    m_integral = 0.0;
    m_last_phase_error = 0.0;
    
    // Initialize sampling state machine
    m_sample_state = SampleState::WAIT_EDGE;
    m_edge_sample = false;
    m_data_sample = false;
    m_prev_data_sample = false;
    m_prev_data_analog = 0.0;
    
    // 整数时基调度状态（timestep 未知，首个 processing 中惰性初始化）
    m_step_index = 0;
    m_edge_ui = 0;
    m_data_ui = 0;
    m_next_edge_step = 0;
    m_next_data_step = 0;
    m_scheduled_qq = 0.0;
    m_ts_cached = 0.0;
    m_last_written_pe = 1.0;
}

// ============================================================================
// 整数时基调度辅助（2026-10 重写）
// ============================================================================

double RxCdrTdf::quantized_phase() const
{
    return std::round(m_phase / m_params.pai.resolution) * m_params.pai.resolution;
}

long long RxCdrTdf::steps_for(double t, long long floor_step) const
{
    long long n;
    if (t <= 0.0) {
        n = 0;
    } else {
        // ceil 对齐步格；微小相对 epsilon 防止整倍数处 FP 进位
        n = static_cast<long long>(std::ceil(t / m_ts_cached - 1e-9));
    }
    return (n < floor_step) ? floor_step : n;
}

void RxCdrTdf::reschedule(double qq)
{
    // 尚未发出的 edge/data 触发按新 qq 重排（floor = 当前步，允许即拍补发，
    // 保证相位回跳时触发不丢失）
    m_next_edge_step = steps_for(m_edge_ui * m_params.ui + qq, m_step_index);
    
    double sp = m_params.sample_point;
    if (sp < 0.0) sp = 0.0;
    if (sp > 1.0) sp = 1.0;
    m_next_data_step = steps_for(m_data_ui * m_params.ui + sp * m_params.ui + qq,
                                 m_step_index);
    
    // 乱序防护：仅当 data 与尚未发出的 edge 属于同一 UI 时，
    // 才要求 data 不早于该 edge。（若 edge-k 已发出、计数器已前进，
    // 则 data-k 与待发的 edge-(k+1) 分属不同 UI，不得比较抬升，
    // 否则 data-k 会被合并进 edge-(k+1) 导致丢失——2026-10 修复）
    if (m_data_ui == m_edge_ui && m_next_data_step < m_next_edge_step) {
        m_next_data_step = m_next_edge_step;
    }
}

// ============================================================================
// Main Processing
// ============================================================================

void RxCdrTdf::processing()
{
    // ========================================================================
    // Step 1: Read sampled value from Sampler (result of previous trigger)
    // ========================================================================
    // Due to TDF delay, we read the sample from the previous trigger here
    double v_in = in.read();
    bool sampled_value = (v_in > 0.5);
    
    // Store sample based on what we were waiting for
    if (m_sample_state == SampleState::WAIT_DATA) {
        // We were waiting for data sample (second trigger in UI)
        m_data_sample = sampled_value;
        
        // =====================================================================
        // Step 2: Bang-Bang Phase Detection (now we have both samples)
        // =====================================================================
        double phase_error = 0.0;
        
        // 跳变检测：相邻数据样本的模拟量幅度差须超过 edge_threshold
        // （低阈值下小幅跳变也应被检测；高阈值下小幅跳变被抑制）
        double analog_delta = std::abs(v_in - m_prev_data_analog);
        if (analog_delta > m_params.pi.edge_threshold) {
            // 显著跳变 - 执行 BBPD
            if (m_edge_sample == m_data_sample) {
                // Edge sample equals new data value → clock is EARLY
                // (we sampled edge after the transition)
                phase_error = -1.0;
            } else {
                // Edge sample equals old data value → clock is LATE
                // (we sampled edge before the transition)
                phase_error = +1.0;
            }
        }
        // No transition: phase_error remains 0 (no update)
        
        // Store for debug
        m_last_phase_error = phase_error;
        
        // =====================================================================
        // Step 3: PI controller update
        // =====================================================================
        // Update integral term
        m_integral += m_params.pi.ki * phase_error;
        
        // Calculate proportional term
        double prop_term = m_params.pi.kp * phase_error;
        
        // Total PI output (in UI units)
        double pi_output = prop_term + m_integral;
        
        // Scale to seconds
        m_phase = pi_output * m_params.ui;
        
        // =====================================================================
        // Step 4: Phase range limiting
        // =====================================================================
        double range = m_params.pai.range;
        m_phase = std::max(-range, std::min(range, m_phase));
        
        // Anti-windup
        if (m_phase >= range || m_phase <= -range) {
            m_integral = m_phase / m_params.ui - prop_term;
        }
        
        // Save current data sample for next comparison
        m_prev_data_sample = m_data_sample;
        m_prev_data_analog = v_in;
        
        // Reset state for next UI
        m_sample_state = SampleState::WAIT_EDGE;
    }
    else if (m_sample_state == SampleState::WAIT_EDGE) {
        // We were waiting for edge sample (first trigger in UI)
        m_edge_sample = sampled_value;
        // Next we wait for data sample
        m_sample_state = SampleState::WAIT_DATA;
    }
    
    // ========================================================================
    // Step 5: 采样触发调度（2026-10 整数时基重写）
    // ========================================================================
    // 语义：edge 触发 @ k*ui + qq；data 触发 @ k*ui + sp*ui + qq。
    // 步格对齐用 ceil，任何 timestep 下每 UI 恰好 2 个触发。
    if (m_ts_cached <= 0.0) {
        m_ts_cached = get_timestep().to_seconds();
        if (m_ts_cached <= 0.0) {
            m_ts_cached = 1e-12;  // 退化保护
        }
        reschedule(m_scheduled_qq);
    }
    
    // PI 更新了量化相位 → 重排尚未发出的触发（floor = 当前步，允许补发）
    double qq = quantized_phase();
    if (qq != m_scheduled_qq) {
        m_scheduled_qq = qq;
        reschedule(qq);
    }
    
    bool edge_trig = false;
    bool data_trig = false;
    
    if (m_step_index >= m_next_edge_step) {
        edge_trig = true;
        ++m_edge_ui;
        m_next_edge_step = steps_for(m_edge_ui * m_params.ui + m_scheduled_qq,
                                     m_step_index + 1);
    }
    if (m_step_index >= m_next_data_step) {
        data_trig = true;
        ++m_data_ui;
        double sp = m_params.sample_point;
        if (sp < 0.0) sp = 0.0;
        if (sp > 1.0) sp = 1.0;
        m_next_data_step = steps_for(
            m_data_ui * m_params.ui + sp * m_params.ui + m_scheduled_qq,
            m_step_index + 1);
    }
    
    // ========================================================================
    // Step 6: Output
    // ========================================================================
    // Output the actual phase (not quantized) for better visualization
    phase_out.write(m_phase);
    sampling_trigger.write(edge_trig || data_trig);
    
    // BBPD 相位误差（秒）→ DE 域（写变更，事件驱动；供 AdaptionDe 监视）
    double pe_sec = m_last_phase_error * m_params.ui;
    if (pe_sec != m_last_written_pe) {
        phase_error_de.write(pe_sec);
        m_last_written_pe = pe_sec;
    }
    
    ++m_step_index;
}

} // namespace serdes
