#include "ams/single_to_diff.h"

namespace serdes {

SingleToDiffTdf::SingleToDiffTdf(sc_core::sc_module_name nm)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out_p("out_p")
    , out_n("out_n")
{
}

void SingleToDiffTdf::set_attributes()
{
    // Accept any timestep from connected modules
    // Timestep will be determined by the cluster
}

void SingleToDiffTdf::processing()
{
    // Read single-ended input
    double input = in.read();
    
    // Convert to differential:
    // out_p = +input (positive terminal)
    // out_n = -input (negative terminal, inverted)
    out_p.write(input);
    out_n.write(-input);
}

} // namespace serdes
