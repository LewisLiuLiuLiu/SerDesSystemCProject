/**
 * @file test_config_loader.cpp
 * @brief ConfigLoader 单元测试：单一事实源 / 覆盖语义 / fail-fast / 可审计
 */

#include <gtest/gtest.h>
#include "de/config_loader.h"
#include "common/parameters.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace serdes;

#ifndef SERDES_CONFIG_DIR
#define SERDES_CONFIG_DIR "config"
#endif

namespace {

std::string tmp_path(const char* name) {
    return std::string("config_loader_test_") + name;
}

bool write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << content;
    return f.good();
}

} // namespace

// 默认值 = 结构体构造默认值（单一事实源）
TEST(ConfigLoaderTest, DefaultEqualsStructDefaults) {
    SystemParams a = ConfigLoader::load_default();
    SystemParams b{};
    EXPECT_DOUBLE_EQ(a.global.Fs, b.global.Fs);
    EXPECT_DOUBLE_EQ(a.global.UI, b.global.UI);
    EXPECT_DOUBLE_EQ(a.global.duration, b.global.duration);
    EXPECT_EQ(a.global.seed, b.global.seed);
    EXPECT_EQ(a.wave.type, b.wave.type);
    EXPECT_EQ(a.tx.ffe.taps, b.tx.ffe.taps);
    EXPECT_DOUBLE_EQ(a.rx.sampler.hysteresis, b.rx.sampler.hysteresis);
    EXPECT_DOUBLE_EQ(a.rx.dfe_summer.ui, b.rx.dfe_summer.ui);
    EXPECT_DOUBLE_EQ(a.cdr.pi.kp, b.cdr.pi.kp);
    EXPECT_EQ(a.clock.type, b.clock.type);
    EXPECT_EQ(a.eye.ui_bins, b.eye.ui_bins);
    EXPECT_TRUE(a.adaption.agc.enabled);
    EXPECT_EQ(a.adaption.dfe.num_taps, b.adaption.dfe.num_taps);
}

// 规范默认配置：加载后与结构体默认值一致（广泛抽查）
TEST(ConfigLoaderTest, LoadDefaultJsonMatchesDefaults) {
    const std::string path = std::string(SERDES_CONFIG_DIR) + "/default.json";
    SystemParams p;
    ASSERT_TRUE(ConfigLoader::loadFromFile(path, p)) << path;
    SystemParams d{};
    EXPECT_DOUBLE_EQ(p.global.Fs, d.global.Fs);
    EXPECT_DOUBLE_EQ(p.global.UI, d.global.UI);
    EXPECT_DOUBLE_EQ(p.global.duration, d.global.duration);
    EXPECT_EQ(p.global.seed, d.global.seed);
    EXPECT_EQ(p.wave.type, d.wave.type);
    EXPECT_EQ(p.wave.poly, d.wave.poly);
    EXPECT_EQ(p.tx.ffe.taps, d.tx.ffe.taps);
    EXPECT_DOUBLE_EQ(p.tx.driver.vswing, d.tx.driver.vswing);
    ASSERT_FALSE(p.tx.driver.poles.empty());
    EXPECT_DOUBLE_EQ(p.tx.driver.poles.at(0), d.tx.driver.poles.at(0));
    EXPECT_DOUBLE_EQ(p.channel.attenuation_db, d.channel.attenuation_db);
    EXPECT_DOUBLE_EQ(p.rx.ctle.dc_gain, d.rx.ctle.dc_gain);
    EXPECT_DOUBLE_EQ(p.rx.vga.dc_gain, d.rx.vga.dc_gain);
    EXPECT_DOUBLE_EQ(p.rx.sampler.hysteresis, d.rx.sampler.hysteresis);
    EXPECT_DOUBLE_EQ(p.rx.sampler.resolution, d.rx.sampler.resolution);
    EXPECT_EQ(p.rx.dfe_summer.tap_coeffs, d.rx.dfe_summer.tap_coeffs);
    EXPECT_DOUBLE_EQ(p.rx.dfe_summer.ui, d.rx.dfe_summer.ui);
    EXPECT_DOUBLE_EQ(p.cdr.pi.kp, d.cdr.pi.kp);
    EXPECT_DOUBLE_EQ(p.cdr.pai.range, d.cdr.pai.range);
    EXPECT_EQ(p.clock.type, d.clock.type);
    EXPECT_DOUBLE_EQ(p.clock.frequency, d.clock.frequency);
    EXPECT_EQ(p.eye.amp_bins, d.eye.amp_bins);
    EXPECT_DOUBLE_EQ(p.adaption.agc.target_amplitude, d.adaption.agc.target_amplitude);
    EXPECT_EQ(p.adaption.dfe.initial_taps, d.adaption.dfe.initial_taps);
    EXPECT_DOUBLE_EQ(p.adaption.vref_adapt.vref_pos, d.adaption.vref_adapt.vref_pos);
    EXPECT_EQ(p.adaption.safety.error_burst_threshold, d.adaption.safety.error_burst_threshold);
}

// 覆盖语义：仅文件出现的键生效，未出现键保持默认
TEST(ConfigLoaderTest, OverrideOnlySpecifiedKeys) {
    const std::string path = tmp_path("override.json");
    ASSERT_TRUE(write_file(path,
        "{ \"global\": {\"Fs\": 1e11}, \"rx\": {\"sampler\": {\"hysteresis\": 0.005}} }"));
    SystemParams p;
    ASSERT_TRUE(ConfigLoader::loadFromFile(path, p));
    SystemParams d{};
    EXPECT_DOUBLE_EQ(p.global.Fs, 1e11);
    EXPECT_DOUBLE_EQ(p.rx.sampler.hysteresis, 0.005);
    EXPECT_DOUBLE_EQ(p.global.UI, d.global.UI);
    EXPECT_DOUBLE_EQ(p.rx.sampler.resolution, d.rx.sampler.resolution);
    std::remove(path.c_str());
}

// 类型错误 → fail-fast
TEST(ConfigLoaderTest, TypeErrorRejected) {
    const std::string path = tmp_path("type_error.json");
    ASSERT_TRUE(write_file(path, "{ \"global\": {\"Fs\": \"fast\"} }"));
    SystemParams p;
    EXPECT_FALSE(ConfigLoader::loadFromFile(path, p));
    std::remove(path.c_str());
}

// 未知键（'\_' 前缀为注释）→ 警告但加载成功，已知键仍生效
TEST(ConfigLoaderTest, UnknownKeyWarnsButLoads) {
    const std::string path = tmp_path("unknown_key.json");
    ASSERT_TRUE(write_file(path,
        "{ \"_comment\": \"note\", \"global\": {\"Fs\": 9e10, \"mystery\": 1} }"));
    SystemParams p;
    EXPECT_TRUE(ConfigLoader::loadFromFile(path, p));
    EXPECT_DOUBLE_EQ(p.global.Fs, 9e10);
    std::remove(path.c_str());
}

// 枚举错误 → 失败
TEST(ConfigLoaderTest, BadEnumRejected) {
    const std::string path = tmp_path("bad_enum.json");
    ASSERT_TRUE(write_file(path, "{ \"wave\": {\"type\": \"PRBS99\"} }"));
    SystemParams p;
    EXPECT_FALSE(ConfigLoader::loadFromFile(path, p));
    std::remove(path.c_str());
}

// 文件不存在 / YAML → 失败
TEST(ConfigLoaderTest, MissingFileRejected) {
    SystemParams p;
    EXPECT_FALSE(ConfigLoader::loadFromFile("no_such_config_file.json", p));
}

TEST(ConfigLoaderTest, YamlRejected) {
    SystemParams p;
    EXPECT_FALSE(ConfigLoader::loadFromFile("whatever_config.yaml", p));
}
