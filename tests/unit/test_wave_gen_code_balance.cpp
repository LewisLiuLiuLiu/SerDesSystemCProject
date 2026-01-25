/**
 * @file test_wave_gen_code_balance.cpp
 * @brief Unit test for WaveGenerationTdf module - Code Balance Verification
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenStatisticsTest, CodeBalanceVerification) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_balance", params, 80e9, 12345, 10000);
    
    sc_core::sc_start(200, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GT(samples.size(), 1000) << "Need enough samples for balance check";
    
    int positive_count = 0;
    int negative_count = 0;
    for (const auto& s : samples) {
        if (s > 0) positive_count++;
        else negative_count++;
    }
    
    double balance = std::abs(positive_count - negative_count) / 
                     static_cast<double>(samples.size());
    EXPECT_LT(balance, 0.1) << "Code balance should be < 10%";
    
    sc_core::sc_stop();
}
