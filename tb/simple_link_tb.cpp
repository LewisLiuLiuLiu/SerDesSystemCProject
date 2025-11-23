#include <systemc>
#include <systemc-ams>
#include "ams/wave_generation.h"
#include "common/parameters.h"

int sc_main(int argc, char* argv[]) {
    using namespace serdes;
    
    // Create default parameters
    WaveGenParams wave_params;
    
    // Create WaveGeneration module
    WaveGenerationTdf wave_gen("wave_gen", wave_params);
    
    // Create trace file
    sca_util::sca_trace_file* tf = sca_util::sca_create_tabular_trace_file("simple_link");
    sca_util::sca_trace(tf, wave_gen.out, "wave_out");
    
    // Run simulation
    sc_core::sc_start(1.0, sc_core::SC_US);
    
    // Close trace file
    sca_util::sca_close_tabular_trace_file(tf);
    
    std::cout << "Simple link testbench completed successfully!" << std::endl;
    
    return 0;
}
