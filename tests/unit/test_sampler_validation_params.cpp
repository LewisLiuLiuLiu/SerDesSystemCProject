/**
 * @file test_sampler_validation_params.cpp
 * @brief RxSamplerTdf 参数校验（2026-10：hysteresis 已废弃、不再校验）
 */

#include "sampler_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(SamplerValidationTest, DeprecatedHysteresisNoLongerValidated) {
    // hysteresis 已从判决路径移除，仅为配置兼容保留：
    // 任意取值（包括 hysteresis >= resolution）都不应导致构造失败。
    RxSamplerParams params;
    params.hysteresis = 0.1;
    params.resolution = 0.05;

    EXPECT_NO_THROW(
        RxSamplerTdf* sampler = new RxSamplerTdf("sampler", params);
        delete sampler;
    ) << "hysteresis is deprecated and must not be validated any more";
}

TEST(SamplerValidationTest, InvalidPhaseSourceStillRejected) {
    RxSamplerParams params;
    params.phase_source = "invalid";

    EXPECT_THROW(
        RxSamplerTdf* sampler = new RxSamplerTdf("sampler", params),
        std::invalid_argument
    ) << "Phase source validation must still be active";
}
