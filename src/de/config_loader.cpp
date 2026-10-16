/**
 * @file config_loader.cpp
 * @brief SystemParams 配置加载器实现
 *
 * 设计原则（2026-10 重构）：
 * - 单一事实源：参数默认值 = parameters.h 中结构体构造默认值，本加载器不再复制默认值。
 * - 文件 = 覆盖层：JSON 以覆盖方式作用于默认值之上；未出现的键保持默认。
 * - 加载即校验：类型/枚举错误 → 返回 false（fail fast，不产生半加载结果）；
 *   未知键 → 警告并忽略；键名以 '_' 开头视为注释，忽略。
 * - 可审计：成功时打印"实际发生变化的键"清单。
 * - 格式：仅 JSON（使用仓库内 vendored third_party/json.hpp）；YAML 不支持并显式报错。
 */

#include "de/config_loader.h"
#include "../third_party/json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace serdes {

namespace {

using nlohmann::json;

// ============================================================================
// 解析上下文
// ============================================================================
struct ParseCtx {
    std::vector<std::string> applied;    ///< 值发生变化的键路径（相对默认值）
    std::vector<std::string> warnings;   ///< 未知键等告警
    std::vector<std::string> errors;     ///< 类型/枚举错误（非空 → 加载失败）
};

std::string join_path(const std::string& parent, const std::string& key) {
    return parent.empty() ? key : parent + "." + key;
}

// 子节获取：存在但类型不对 → 记录错误
const json* get_section(ParseCtx& ctx, const json& parent, const char* key, const std::string& parent_path) {
    auto it = parent.find(key);
    if (it == parent.end()) return nullptr;
    if (!it->is_object()) {
        ctx.errors.push_back(join_path(parent_path, key) + ": 期望对象（object）");
        return nullptr;
    }
    return &(*it);
}

// ---- 字段读取辅助（未出现=跳过；类型错=错误；值变化=记录 applied） ----
bool num_field(ParseCtx& ctx, const json& sec, const char* key, const std::string& parent, double& target) {
    auto it = sec.find(key);
    if (it == sec.end()) return false;
    const std::string full = join_path(parent, key);
    if (!it->is_number()) { ctx.errors.push_back(full + ": 期望数字"); return false; }
    double v = it->get<double>();
    if (v != target) ctx.applied.push_back(full);
    target = v;
    return true;
}

bool int_field(ParseCtx& ctx, const json& sec, const char* key, const std::string& parent, int& target) {
    auto it = sec.find(key);
    if (it == sec.end()) return false;
    const std::string full = join_path(parent, key);
    if (!it->is_number()) { ctx.errors.push_back(full + ": 期望整数"); return false; }
    double v = it->get<double>();
    if (v != static_cast<double>(static_cast<int>(v))) { ctx.errors.push_back(full + ": 期望整数"); return false; }
    int iv = static_cast<int>(v);
    if (iv != target) ctx.applied.push_back(full);
    target = iv;
    return true;
}

bool uint_field(ParseCtx& ctx, const json& sec, const char* key, const std::string& parent, unsigned int& target) {
    auto it = sec.find(key);
    if (it == sec.end()) return false;
    const std::string full = join_path(parent, key);
    if (!it->is_number()) { ctx.errors.push_back(full + ": 期望无符号整数"); return false; }
    double v = it->get<double>();
    if (v < 0.0 || v != static_cast<double>(static_cast<unsigned int>(v))) {
        ctx.errors.push_back(full + ": 期望无符号整数");
        return false;
    }
    unsigned int uv = static_cast<unsigned int>(v);
    if (uv != target) ctx.applied.push_back(full);
    target = uv;
    return true;
}

bool bool_field(ParseCtx& ctx, const json& sec, const char* key, const std::string& parent, bool& target) {
    auto it = sec.find(key);
    if (it == sec.end()) return false;
    const std::string full = join_path(parent, key);
    if (!it->is_boolean()) { ctx.errors.push_back(full + ": 期望布尔值"); return false; }
    bool v = it->get<bool>();
    if (v != target) ctx.applied.push_back(full);
    target = v;
    return true;
}

bool str_field(ParseCtx& ctx, const json& sec, const char* key, const std::string& parent, std::string& target) {
    auto it = sec.find(key);
    if (it == sec.end()) return false;
    const std::string full = join_path(parent, key);
    if (!it->is_string()) { ctx.errors.push_back(full + ": 期望字符串"); return false; }
    std::string v = it->get<std::string>();
    if (v != target) ctx.applied.push_back(full);
    target = v;
    return true;
}

bool num_array_field(ParseCtx& ctx, const json& sec, const char* key, const std::string& parent,
                     std::vector<double>& target) {
    auto it = sec.find(key);
    if (it == sec.end()) return false;
    const std::string full = join_path(parent, key);
    if (!it->is_array()) { ctx.errors.push_back(full + ": 期望数组"); return false; }
    std::vector<double> v;
    v.reserve(it->size());
    for (const auto& e : *it) {
        if (!e.is_number()) { ctx.errors.push_back(full + ": 数组元素必须是数字"); return false; }
        v.push_back(e.get<double>());
    }
    if (v != target) ctx.applied.push_back(full);
    target = v;
    return true;
}

bool prbs_field(ParseCtx& ctx, const json& sec, const char* key, const std::string& parent, PRBSType& target) {
    auto it = sec.find(key);
    if (it == sec.end()) return false;
    const std::string full = join_path(parent, key);
    if (!it->is_string()) { ctx.errors.push_back(full + ": 期望字符串（PRBS 类型）"); return false; }
    const std::string s = it->get<std::string>();
    PRBSType t;
    if (s == "PRBS7")       t = PRBSType::PRBS7;
    else if (s == "PRBS9")  t = PRBSType::PRBS9;
    else if (s == "PRBS15") t = PRBSType::PRBS15;
    else if (s == "PRBS23") t = PRBSType::PRBS23;
    else if (s == "PRBS31") t = PRBSType::PRBS31;
    else if (s == "CUSTOM") t = PRBSType::CUSTOM;
    else { ctx.errors.push_back(full + ": 未知 PRBS 类型 '" + s + "'"); return false; }
    if (t != target) ctx.applied.push_back(full);
    target = t;
    return true;
}

bool clock_type_field(ParseCtx& ctx, const json& sec, const char* key, const std::string& parent, ClockType& target) {
    auto it = sec.find(key);
    if (it == sec.end()) return false;
    const std::string full = join_path(parent, key);
    if (!it->is_string()) { ctx.errors.push_back(full + ": 期望字符串（时钟类型）"); return false; }
    const std::string s = it->get<std::string>();
    ClockType t;
    if (s == "IDEAL")      t = ClockType::IDEAL;
    else if (s == "PLL")   t = ClockType::PLL;
    else if (s == "ADPLL") t = ClockType::ADPLL;
    else { ctx.errors.push_back(full + ": 未知时钟类型 '" + s + "'"); return false; }
    if (t != target) ctx.applied.push_back(full);
    target = t;
    return true;
}

// 未知键检查（'_' 前缀视为注释，静默忽略）
void check_known(ParseCtx& ctx, const json& sec, const std::string& path,
                 std::initializer_list<const char*> known) {
    for (auto it = sec.begin(); it != sec.end(); ++it) {
        const std::string& k = it.key();
        if (!k.empty() && k[0] == '_') continue;
        bool ok = false;
        for (auto n : known) {
            if (k == n) { ok = true; break; }
        }
        if (!ok) ctx.warnings.push_back("未知键（已忽略）: " + join_path(path, k));
    }
}

// ============================================================================
// 各节解析
// ============================================================================
void parse_jitter(ParseCtx& c, const json& j, const std::string& p, JitterParams& v) {
    check_known(c, j, p, {"RJ_sigma", "SJ_freq", "SJ_pp"});
    num_field(c, j, "RJ_sigma", p, v.RJ_sigma);
    num_array_field(c, j, "SJ_freq", p, v.SJ_freq);
    num_array_field(c, j, "SJ_pp", p, v.SJ_pp);
}

void parse_modulation(ParseCtx& c, const json& j, const std::string& p, ModulationParams& v) {
    check_known(c, j, p, {"AM", "PM"});
    num_field(c, j, "AM", p, v.AM);
    num_field(c, j, "PM", p, v.PM);
}

void parse_wave(ParseCtx& c, const json& j, const std::string& p, WaveGenParams& v) {
    check_known(c, j, p, {"type", "poly", "init", "single_pulse", "jitter", "modulation"});
    prbs_field(c, j, "type", p, v.type);
    str_field(c, j, "poly", p, v.poly);
    str_field(c, j, "init", p, v.init);
    num_field(c, j, "single_pulse", p, v.single_pulse);
    if (const json* sub = get_section(c, j, "jitter", p)) parse_jitter(c, *sub, p + ".jitter", v.jitter);
    if (const json* sub = get_section(c, j, "modulation", p)) parse_modulation(c, *sub, p + ".modulation", v.modulation);
}

void parse_driver_psrr(ParseCtx& c, const json& j, const std::string& p, TxDriverParams::PsrrParams& v) {
    check_known(c, j, p, {"enable", "gain", "poles", "vdd_nom"});
    bool_field(c, j, "enable", p, v.enable);
    num_field(c, j, "gain", p, v.gain);
    num_array_field(c, j, "poles", p, v.poles);
    num_field(c, j, "vdd_nom", p, v.vdd_nom);
}

void parse_tx_driver(ParseCtx& c, const json& j, const std::string& p, TxDriverParams& v) {
    check_known(c, j, p, {"dc_gain", "vswing", "vcm_out", "output_impedance", "poles",
                          "sat_mode", "vlin", "psrr", "imbalance", "slew_rate"});
    num_field(c, j, "dc_gain", p, v.dc_gain);
    num_field(c, j, "vswing", p, v.vswing);
    num_field(c, j, "vcm_out", p, v.vcm_out);
    num_field(c, j, "output_impedance", p, v.output_impedance);
    num_array_field(c, j, "poles", p, v.poles);
    str_field(c, j, "sat_mode", p, v.sat_mode);
    num_field(c, j, "vlin", p, v.vlin);
    if (const json* sub = get_section(c, j, "psrr", p)) parse_driver_psrr(c, *sub, p + ".psrr", v.psrr);
    if (const json* sub = get_section(c, j, "imbalance", p)) {
        check_known(c, *sub, p + ".imbalance", {"gain_mismatch", "skew"});
        num_field(c, *sub, "gain_mismatch", p + ".imbalance", v.imbalance.gain_mismatch);
        num_field(c, *sub, "skew", p + ".imbalance", v.imbalance.skew);
    }
    if (const json* sub = get_section(c, j, "slew_rate", p)) {
        check_known(c, *sub, p + ".slew_rate", {"enable", "max_slew_rate"});
        bool_field(c, *sub, "enable", p + ".slew_rate", v.slew_rate.enable);
        num_field(c, *sub, "max_slew_rate", p + ".slew_rate", v.slew_rate.max_slew_rate);
    }
}

void parse_tx(ParseCtx& c, const json& j, const std::string& p, TxParams& v) {
    check_known(c, j, p, {"ffe", "mux_lane", "driver"});
    if (const json* sub = get_section(c, j, "ffe", p)) {
        check_known(c, *sub, p + ".ffe", {"taps"});
        num_array_field(c, *sub, "taps", p + ".ffe", v.ffe.taps);
    }
    int_field(c, j, "mux_lane", p, v.mux_lane);
    if (const json* sub = get_section(c, j, "driver", p)) parse_tx_driver(c, *sub, p + ".driver", v.driver);
}

void parse_channel(ParseCtx& c, const json& j, const std::string& p, ChannelParams& v) {
    check_known(c, j, p, {"touchstone", "ports", "crosstalk", "bidirectional",
                          "attenuation_db", "bandwidth_hz"});
    str_field(c, j, "touchstone", p, v.touchstone);
    int_field(c, j, "ports", p, v.ports);
    bool_field(c, j, "crosstalk", p, v.crosstalk);
    bool_field(c, j, "bidirectional", p, v.bidirectional);
    num_field(c, j, "attenuation_db", p, v.attenuation_db);
    num_field(c, j, "bandwidth_hz", p, v.bandwidth_hz);
}

template <typename S>
void parse_amp_psrr(ParseCtx& c, const json& j, const std::string& p, S& v) {
    check_known(c, j, p, {"enable", "gain", "zeros", "poles", "vdd_nom"});
    bool_field(c, j, "enable", p, v.enable);
    num_field(c, j, "gain", p, v.gain);
    num_array_field(c, j, "zeros", p, v.zeros);
    num_array_field(c, j, "poles", p, v.poles);
    num_field(c, j, "vdd_nom", p, v.vdd_nom);
}

template <typename S>
void parse_amp_cmfb(ParseCtx& c, const json& j, const std::string& p, S& v) {
    check_known(c, j, p, {"enable", "bandwidth", "loop_gain"});
    bool_field(c, j, "enable", p, v.enable);
    num_field(c, j, "bandwidth", p, v.bandwidth);
    num_field(c, j, "loop_gain", p, v.loop_gain);
}

template <typename S>
void parse_amp_cmrr(ParseCtx& c, const json& j, const std::string& p, S& v) {
    check_known(c, j, p, {"enable", "gain", "zeros", "poles"});
    bool_field(c, j, "enable", p, v.enable);
    num_field(c, j, "gain", p, v.gain);
    num_array_field(c, j, "zeros", p, v.zeros);
    num_array_field(c, j, "poles", p, v.poles);
}

// CTLE 与 VGA 参数字段同名同型，模板复用
template <typename P>
void parse_rx_amp(ParseCtx& c, const json& j, const std::string& p, P& v) {
    check_known(c, j, p, {"zeros", "poles", "dc_gain", "vcm_out", "offset_enable", "vos",
                          "noise_enable", "vnoise_sigma", "sat_min", "sat_max",
                          "psrr", "cmfb", "cmrr"});
    num_array_field(c, j, "zeros", p, v.zeros);
    num_array_field(c, j, "poles", p, v.poles);
    num_field(c, j, "dc_gain", p, v.dc_gain);
    num_field(c, j, "vcm_out", p, v.vcm_out);
    bool_field(c, j, "offset_enable", p, v.offset_enable);
    num_field(c, j, "vos", p, v.vos);
    bool_field(c, j, "noise_enable", p, v.noise_enable);
    num_field(c, j, "vnoise_sigma", p, v.vnoise_sigma);
    num_field(c, j, "sat_min", p, v.sat_min);
    num_field(c, j, "sat_max", p, v.sat_max);
    if (const json* sub = get_section(c, j, "psrr", p)) parse_amp_psrr(c, *sub, p + ".psrr", v.psrr);
    if (const json* sub = get_section(c, j, "cmfb", p)) parse_amp_cmfb(c, *sub, p + ".cmfb", v.cmfb);
    if (const json* sub = get_section(c, j, "cmrr", p)) parse_amp_cmrr(c, *sub, p + ".cmrr", v.cmrr);
}

void parse_sampler(ParseCtx& c, const json& j, const std::string& p, RxSamplerParams& v) {
    check_known(c, j, p, {"threshold", "hysteresis", "sample_delay", "phase_source", "resolution",
                          "offset_enable", "offset_value", "noise_enable", "noise_sigma", "noise_seed"});
    num_field(c, j, "threshold", p, v.threshold);
    num_field(c, j, "hysteresis", p, v.hysteresis);
    num_field(c, j, "sample_delay", p, v.sample_delay);
    str_field(c, j, "phase_source", p, v.phase_source);
    num_field(c, j, "resolution", p, v.resolution);
    bool_field(c, j, "offset_enable", p, v.offset_enable);
    num_field(c, j, "offset_value", p, v.offset_value);
    bool_field(c, j, "noise_enable", p, v.noise_enable);
    num_field(c, j, "noise_sigma", p, v.noise_sigma);
    uint_field(c, j, "noise_seed", p, v.noise_seed);
}

void parse_dfe_summer(ParseCtx& c, const json& j, const std::string& p, RxDfeSummerParams& v) {
    check_known(c, j, p, {"tap_coeffs", "ui", "vcm_out", "vtap", "map_mode", "enable",
                          "sat_enable", "sat_min", "sat_max"});
    num_array_field(c, j, "tap_coeffs", p, v.tap_coeffs);
    num_field(c, j, "ui", p, v.ui);
    num_field(c, j, "vcm_out", p, v.vcm_out);
    num_field(c, j, "vtap", p, v.vtap);
    str_field(c, j, "map_mode", p, v.map_mode);
    bool_field(c, j, "enable", p, v.enable);
    bool_field(c, j, "sat_enable", p, v.sat_enable);
    num_field(c, j, "sat_min", p, v.sat_min);
    num_field(c, j, "sat_max", p, v.sat_max);
}

void parse_cdr(ParseCtx& c, const json& j, const std::string& p, CdrParams& v) {
    check_known(c, j, p, {"pi", "pai", "ui", "sample_point", "debug_enable"});
    if (const json* sub = get_section(c, j, "pi", p)) {
        check_known(c, *sub, p + ".pi", {"kp", "ki", "edge_threshold", "adaptive_threshold"});
        num_field(c, *sub, "kp", p + ".pi", v.pi.kp);
        num_field(c, *sub, "ki", p + ".pi", v.pi.ki);
        num_field(c, *sub, "edge_threshold", p + ".pi", v.pi.edge_threshold);
        bool_field(c, *sub, "adaptive_threshold", p + ".pi", v.pi.adaptive_threshold);
    }
    if (const json* sub = get_section(c, j, "pai", p)) {
        check_known(c, *sub, p + ".pai", {"resolution", "range"});
        num_field(c, *sub, "resolution", p + ".pai", v.pai.resolution);
        num_field(c, *sub, "range", p + ".pai", v.pai.range);
    }
    num_field(c, j, "ui", p, v.ui);
    num_field(c, j, "sample_point", p, v.sample_point);
    bool_field(c, j, "debug_enable", p, v.debug_enable);
}

void parse_rx(ParseCtx& c, const json& j, const std::string& p, RxParams& v) {
    check_known(c, j, p, {"ctle", "vga", "sampler", "dfe_summer", "cdr"});
    if (const json* sub = get_section(c, j, "ctle", p)) parse_rx_amp(c, *sub, p + ".ctle", v.ctle);
    if (const json* sub = get_section(c, j, "vga", p)) parse_rx_amp(c, *sub, p + ".vga", v.vga);
    if (const json* sub = get_section(c, j, "sampler", p)) parse_sampler(c, *sub, p + ".sampler", v.sampler);
    if (const json* sub = get_section(c, j, "dfe_summer", p)) parse_dfe_summer(c, *sub, p + ".dfe_summer", v.dfe_summer);
    // rx.cdr 与顶层 cdr 为历史双份字段；两者都写时以顶层为准（parse_root 内同步）
    if (const json* sub = get_section(c, j, "cdr", p)) parse_cdr(c, *sub, p + ".cdr", v.cdr);
}

void parse_clock(ParseCtx& c, const json& j, const std::string& p, ClockParams& v) {
    check_known(c, j, p, {"type", "frequency", "pll"});
    clock_type_field(c, j, "type", p, v.type);
    num_field(c, j, "frequency", p, v.frequency);
    if (const json* sub = get_section(c, j, "pll", p)) {
        check_known(c, *sub, p + ".pll", {"pd_type", "cp_current", "lf_R", "lf_C",
                                         "vco_Kvco", "vco_f0", "divider"});
        str_field(c, *sub, "pd_type", p + ".pll", v.pll.pd_type);
        num_field(c, *sub, "cp_current", p + ".pll", v.pll.cp_current);
        num_field(c, *sub, "lf_R", p + ".pll", v.pll.lf_R);
        num_field(c, *sub, "lf_C", p + ".pll", v.pll.lf_C);
        num_field(c, *sub, "vco_Kvco", p + ".pll", v.pll.vco_Kvco);
        num_field(c, *sub, "vco_f0", p + ".pll", v.pll.vco_f0);
        int_field(c, *sub, "divider", p + ".pll", v.pll.divider);
    }
}

void parse_eye(ParseCtx& c, const json& j, const std::string& p, EyeParams& v) {
    check_known(c, j, p, {"ui_bins", "amp_bins", "measure_length"});
    int_field(c, j, "ui_bins", p, v.ui_bins);
    int_field(c, j, "amp_bins", p, v.amp_bins);
    num_field(c, j, "measure_length", p, v.measure_length);
}

void parse_adaption(ParseCtx& c, const json& j, const std::string& p, AdaptionParams& v) {
    check_known(c, j, p, {"Fs", "UI", "seed", "update_mode", "fast_update_period",
                          "slow_update_period", "agc", "dfe", "vref_adapt", "cdr_pi", "safety"});
    num_field(c, j, "Fs", p, v.Fs);
    num_field(c, j, "UI", p, v.UI);
    uint_field(c, j, "seed", p, v.seed);
    str_field(c, j, "update_mode", p, v.update_mode);
    num_field(c, j, "fast_update_period", p, v.fast_update_period);
    num_field(c, j, "slow_update_period", p, v.slow_update_period);

    if (const json* sub = get_section(c, j, "agc", p)) {
        check_known(c, *sub, p + ".agc", {"enabled", "target_amplitude", "kp", "ki",
                                         "gain_min", "gain_max", "rate_limit", "initial_gain"});
        bool_field(c, *sub, "enabled", p + ".agc", v.agc.enabled);
        num_field(c, *sub, "target_amplitude", p + ".agc", v.agc.target_amplitude);
        num_field(c, *sub, "kp", p + ".agc", v.agc.kp);
        num_field(c, *sub, "ki", p + ".agc", v.agc.ki);
        num_field(c, *sub, "gain_min", p + ".agc", v.agc.gain_min);
        num_field(c, *sub, "gain_max", p + ".agc", v.agc.gain_max);
        num_field(c, *sub, "rate_limit", p + ".agc", v.agc.rate_limit);
        num_field(c, *sub, "initial_gain", p + ".agc", v.agc.initial_gain);
    }
    if (const json* sub = get_section(c, j, "dfe", p)) {
        check_known(c, *sub, p + ".dfe", {"enabled", "num_taps", "algorithm", "mu", "leakage",
                                         "initial_taps", "tap_min", "tap_max", "stats_period",
                                         "mu_startup", "mu_acquire", "mu_track"});
        bool_field(c, *sub, "enabled", p + ".dfe", v.dfe.enabled);
        int_field(c, *sub, "num_taps", p + ".dfe", v.dfe.num_taps);
        str_field(c, *sub, "algorithm", p + ".dfe", v.dfe.algorithm);
        num_field(c, *sub, "mu", p + ".dfe", v.dfe.mu);
        num_field(c, *sub, "leakage", p + ".dfe", v.dfe.leakage);
        num_array_field(c, *sub, "initial_taps", p + ".dfe", v.dfe.initial_taps);
        num_field(c, *sub, "tap_min", p + ".dfe", v.dfe.tap_min);
        num_field(c, *sub, "tap_max", p + ".dfe", v.dfe.tap_max);
        int_field(c, *sub, "stats_period", p + ".dfe", v.dfe.stats_period);
        num_field(c, *sub, "mu_startup", p + ".dfe", v.dfe.mu_startup);
        num_field(c, *sub, "mu_acquire", p + ".dfe", v.dfe.mu_acquire);
        num_field(c, *sub, "mu_track", p + ".dfe", v.dfe.mu_track);
    }
    if (const json* sub = get_section(c, j, "vref_adapt", p)) {
        check_known(c, *sub, p + ".vref_adapt", {"enabled", "vref_pos", "vref_neg", "vref_initial",
                                                 "target_confidence", "asymmetry_alert", "adapt_alpha"});
        bool_field(c, *sub, "enabled", p + ".vref_adapt", v.vref_adapt.enabled);
        num_field(c, *sub, "vref_pos", p + ".vref_adapt", v.vref_adapt.vref_pos);
        num_field(c, *sub, "vref_neg", p + ".vref_adapt", v.vref_adapt.vref_neg);
        num_field(c, *sub, "vref_initial", p + ".vref_adapt", v.vref_adapt.vref_initial);
        num_field(c, *sub, "target_confidence", p + ".vref_adapt", v.vref_adapt.target_confidence);
        num_field(c, *sub, "asymmetry_alert", p + ".vref_adapt", v.vref_adapt.asymmetry_alert);
        num_field(c, *sub, "adapt_alpha", p + ".vref_adapt", v.vref_adapt.adapt_alpha);
    }
    if (const json* sub = get_section(c, j, "cdr_pi", p)) {
        check_known(c, *sub, p + ".cdr_pi", {"enabled", "kp", "ki", "phase_resolution", "phase_range",
                                            "anti_windup", "initial_phase"});
        bool_field(c, *sub, "enabled", p + ".cdr_pi", v.cdr_pi.enabled);
        num_field(c, *sub, "kp", p + ".cdr_pi", v.cdr_pi.kp);
        num_field(c, *sub, "ki", p + ".cdr_pi", v.cdr_pi.ki);
        num_field(c, *sub, "phase_resolution", p + ".cdr_pi", v.cdr_pi.phase_resolution);
        num_field(c, *sub, "phase_range", p + ".cdr_pi", v.cdr_pi.phase_range);
        bool_field(c, *sub, "anti_windup", p + ".cdr_pi", v.cdr_pi.anti_windup);
        num_field(c, *sub, "initial_phase", p + ".cdr_pi", v.cdr_pi.initial_phase);
    }
    if (const json* sub = get_section(c, j, "safety", p)) {
        check_known(c, *sub, p + ".safety", {"freeze_on_error", "rollback_enable",
                                             "snapshot_interval", "error_burst_threshold"});
        bool_field(c, *sub, "freeze_on_error", p + ".safety", v.safety.freeze_on_error);
        bool_field(c, *sub, "rollback_enable", p + ".safety", v.safety.rollback_enable);
        num_field(c, *sub, "snapshot_interval", p + ".safety", v.safety.snapshot_interval);
        int_field(c, *sub, "error_burst_threshold", p + ".safety", v.safety.error_burst_threshold);
    }
}

void parse_root(ParseCtx& c, const json& j, SystemParams& v) {
    check_known(c, j, "", {"global", "wave", "tx", "channel", "rx", "cdr", "clock", "eye", "adaption"});
    if (const json* sub = get_section(c, j, "global", "")) {
        check_known(c, *sub, "global", {"Fs", "UI", "duration", "seed"});
        num_field(c, *sub, "Fs", "global", v.global.Fs);
        num_field(c, *sub, "UI", "global", v.global.UI);
        num_field(c, *sub, "duration", "global", v.global.duration);
        uint_field(c, *sub, "seed", "global", v.global.seed);
    }
    if (const json* sub = get_section(c, j, "wave", "")) parse_wave(c, *sub, "wave", v.wave);
    if (const json* sub = get_section(c, j, "tx", "")) parse_tx(c, *sub, "tx", v.tx);
    if (const json* sub = get_section(c, j, "channel", "")) parse_channel(c, *sub, "channel", v.channel);
    if (const json* sub = get_section(c, j, "rx", "")) parse_rx(c, *sub, "rx", v.rx);
    if (const json* sub = get_section(c, j, "cdr", "")) {
        parse_cdr(c, *sub, "cdr", v.cdr);
        // 历史双份字段同步（顶层 cdr 为准）
        v.rx.cdr = v.cdr;
    }
    if (const json* sub = get_section(c, j, "clock", "")) parse_clock(c, *sub, "clock", v.clock);
    if (const json* sub = get_section(c, j, "eye", "")) parse_eye(c, *sub, "eye", v.eye);
    if (const json* sub = get_section(c, j, "adaption", "")) parse_adaption(c, *sub, "adaption", v.adaption);
}

} // namespace

// ============================================================================
// 公开接口
// ============================================================================

bool ConfigLoader::loadFromFile(const std::string& filepath, SystemParams& params) {
    // 检测文件格式
    auto ends_with = [&filepath](const char* suffix) {
        const std::size_t n = std::string(suffix).size();
        return filepath.size() >= n && filepath.compare(filepath.size() - n, n, suffix) == 0;
    };
    if (ends_with(".json")) {
        return loadJSON(filepath, params);
    }
    if (ends_with(".yaml") || ends_with(".yml")) {
        std::cerr << "[Config] YAML 不受支持（未引入 YAML 解析依赖），请改用 JSON: " << filepath << std::endl;
        return false;
    }
    std::cerr << "[Config] 未知配置格式（仅支持 .json）: " << filepath << std::endl;
    return false;
}

bool ConfigLoader::loadJSON(const std::string& filepath, SystemParams& params) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "[Config] 无法打开配置文件: " << filepath << std::endl;
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cerr << "[Config] JSON 解析失败: " << filepath << "\n         " << e.what() << std::endl;
        return false;
    }
    if (!j.is_object()) {
        std::cerr << "[Config] 顶层必须是 JSON 对象: " << filepath << std::endl;
        return false;
    }

    // 默认值 → 文件覆盖（fail-fast：任何类型/枚举错误都不产生半加载结果）
    SystemParams p;   // 结构体构造默认值 = 单一事实源
    ParseCtx ctx;
    parse_root(ctx, j, p);

    for (const auto& w : ctx.warnings) {
        std::cerr << "[Config][warn] " << w << std::endl;
    }
    if (!ctx.errors.empty()) {
        for (const auto& e : ctx.errors) {
            std::cerr << "[Config][error] " << e << std::endl;
        }
        std::cerr << "[Config] 加载失败（" << ctx.errors.size() << " 项类型/枚举错误）: " << filepath << std::endl;
        return false;
    }

    params = p;

    std::cout << "[Config] 已加载: " << filepath
              << "（相对默认值变化 " << ctx.applied.size() << " 项）";
    if (!ctx.applied.empty()) {
        std::cout << ":" << std::endl;
        for (const auto& a : ctx.applied) {
            std::cout << "  - " << a << std::endl;
        }
    } else {
        std::cout << std::endl;
    }
    return true;
}

bool ConfigLoader::loadYAML(const std::string& filepath, SystemParams& params) {
    (void)params;
    // 未引入 YAML 解析依赖；显式失败，避免此前的"假装成功"
    std::cerr << "[Config] YAML 不受支持，请改用 JSON: " << filepath << std::endl;
    return false;
}

SystemParams ConfigLoader::load_default() {
    // 单一事实源：与 parameters.h 结构体构造默认值逐字段一致（不再手工复制）
    return SystemParams{};
}

} // namespace serdes
