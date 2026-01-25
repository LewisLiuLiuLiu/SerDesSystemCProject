/**
 * @file test_wave_gen_repro_run2.cpp
 * @brief Wave Generation reproducibility test - Run 2
 * 
 * This test is part of a pair (run1/run2) that together verify
 * the same seed produces identical sequences across runs.
 * Must be run independently due to SystemC E529 limitation.
 */

#include "wave_generation_test_common.h"
#include <iostream>
#include <iomanip>

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenReproRun2, GenerateSamples) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    const unsigned int SEED = 12345;  // Same seed as run1
    WaveGenTestbench* tb = new WaveGenTestbench("tb_repro_run2", params, 80e9, SEED, 100);
    
    sc_core::sc_start(10, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    unsigned int lfsr_state = tb->get_lfsr_state();
    
    ASSERT_GT(samples.size(), 0) << "Should have generated samples";
    
    // Output all samples for comparison
    // Format: REPRO_RESULT:<seed>:<lfsr_state>:<sample_count>
    // Then one line per sample: SAMPLE:<index>:<value>
    std::cout << "REPRO_RESULT:" << SEED << ":" << lfsr_state << ":" 
              << samples.size() << std::endl;
    
    for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << "SAMPLE:" << i << ":" << std::fixed << std::setprecision(1) 
                  << samples[i] << std::endl;
    }
    
    // Basic validation
    for (size_t i = 0; i < samples.size(); ++i) {
        EXPECT_TRUE(samples[i] == 1.0 || samples[i] == -1.0)
            << "Sample " << i << " should be +1.0 or -1.0";
    }
    
    sc_core::sc_stop();
}
