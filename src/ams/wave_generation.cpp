#include "ams/wave_generation.h"
#include <cmath>
#include <random>

namespace serdes {

// PRBS 多项式配置表
struct PRBSConfig {
    int length;
    unsigned int mask;
    int tap1;
    int tap2;
    unsigned int default_init;
};

static const PRBSConfig PRBS_CONFIGS[] = {
    {7,  0x7F,       6,  5, 0x7F},       // PRBS7
    {9,  0x1FF,      8,  4, 0x1FF},      // PRBS9
    {15, 0x7FFF,     14, 13, 0x7FFF},    // PRBS15
    {23, 0x7FFFFF,   22, 17, 0x7FFFFF},  // PRBS23
    {31, 0x7FFFFFFF, 30, 27, 0x7FFFFFFF} // PRBS31
};

WaveGenerationTdf::WaveGenerationTdf(sc_core::sc_module_name nm, const WaveGenParams& params)
    : sca_tdf::sca_module(nm)
    , out("out")
    , m_params(params)
    , m_lfsr_state(0x7FFFFFFF)
    , m_sample_rate(80e9)
    , m_time(0.0)
{
    // 根据 PRBS 类型设置初始状态
    int prbs_index = static_cast<int>(params.type);
    if (prbs_index >= 0 && prbs_index < 5) {
        m_lfsr_state = PRBS_CONFIGS[prbs_index].default_init;
    }
    
    // 初始化随机数生成器（用于抖动）
    m_rng.seed(12345); // 可以从 GlobalParams 传入
}

void WaveGenerationTdf::set_attributes() {
    out.set_rate(1);
    out.set_timestep(1.0 / m_sample_rate, sc_core::SC_SEC);
}

void WaveGenerationTdf::processing() {
    // 根据 PRBS 类型生成下一个比特
    int prbs_index = static_cast<int>(m_params.type);
    unsigned int feedback = 0;
    
    if (prbs_index >= 0 && prbs_index < 5) {
        const PRBSConfig& config = PRBS_CONFIGS[prbs_index];
        // 计算反馈位
        feedback = ((m_lfsr_state >> config.tap1) ^ (m_lfsr_state >> config.tap2)) & 0x1;
        // 移位并插入反馈
        m_lfsr_state = ((m_lfsr_state << 1) | feedback) & config.mask;
    } else {
        // 默认使用 PRBS31
        feedback = ((m_lfsr_state >> 30) ^ (m_lfsr_state >> 27)) & 0x1;
        m_lfsr_state = ((m_lfsr_state << 1) | feedback) & 0x7FFFFFFF;
    }
    
    // 提取输出比特（最低位）
    bool bit = (m_lfsr_state & 0x1);
    
    // NRZ 调制：比特 0 → -1.0 V，比特 1 → +1.0 V
    double bit_value = bit ? 1.0 : -1.0;
    
    // 抖动注入（简化实现）
    double jitter_offset = 0.0;
    
    // 随机抖动 (RJ)
    if (m_params.jitter.RJ_sigma > 0.0) {
        std::normal_distribution<double> dist(0.0, m_params.jitter.RJ_sigma);
        jitter_offset += dist(m_rng);
    }
    
    // 周期性抖动 (SJ) - 简化，不实际影响时间，只是演示
    for (size_t i = 0; i < m_params.jitter.SJ_freq.size() && i < m_params.jitter.SJ_pp.size(); ++i) {
        double sj_phase = 2.0 * M_PI * m_params.jitter.SJ_freq[i] * m_time;
        jitter_offset += m_params.jitter.SJ_pp[i] * std::sin(sj_phase);
    }
    
    // 注意：在 TDF 域中，真正的抖动应该通过调整采样时刺实现
    // 这里的实现是简化版本，主要用于演示流程
    
    // 输出
    out.write(bit_value);
    
    // 更新时间
    m_time += 1.0 / m_sample_rate;
}

} // namespace serdes
