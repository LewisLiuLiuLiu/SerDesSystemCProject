/**
 * @file test_wave_gen_seed_run1.cpp
 * @brief Wave Generation seed test - Run 1 with seed 12345
 * 
 * This test is part of a pair (run1/run2) that together verify
 * different seeds produce different sequences.
 * Must be run independently due to SystemC E529 limitation.
 */

#include "wave_generation_test_common.h"
#include <iostream>
#include <iomanip>

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenSeedRun1, GenerateWithSeed12345) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    const unsigned int SEED = 12345;
    WaveGenTestbench* tb = new WaveGenTestbench("tb_seed_run1", params, 80e9, SEED, 100);
    
    sc_core::sc_start(10, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    unsigned int lfsr_state = tb->get_lfsr_state();
    
    // Output results for script comparison
    // Format: SEED_RESULT:<seed>:<lfsr_state>:<sample_count>:<first_10_samples_hash>
    ASSERT_GT(samples.size(), 10) << "Should have at least 10 samples";
    
    // Calculate simple hash of first 10 samples for comparison
    long hash = 0;
    for (size_t i = 0; i < 10 && i < samples.size(); ++i) {
        hash = hash * 31 + static_cast<long>(samples[i] > 0 ? 1 : 0);
    }
    
    std::cout << "SEED_RESULT:" << SEED << ":" << lfsr_state << ":" 
              << samples.size() << ":" << hash << std::endl;
    
    // Basic validation
    EXPECT_NE(lfsr_state, 0) << "LFSR state should not be zero";
    EXPECT_NE(lfsr_state, 0x7FFFFFFF) << "LFSR state should have evolved";
    
    for (size_t i = 0; i < samples.size(); ++i) {
        EXPECT_TRUE(samples[i] == 1.0 || samples[i] == -1.0)
            << "Sample " << i << " should be +1.0 or -1.0";
    }
    
    sc_core::sc_stop();
}
