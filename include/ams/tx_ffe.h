#ifndef SERDES_TX_FFE_H
#define SERDES_TX_FFE_H

#include <systemc-ams>
#include "common/parameters.h"
#include <vector>

namespace serdes {

class TxFfeTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    
    TxFfeTdf(sc_core::sc_module_name nm, const TxFfeParams& params);
    
    void set_attributes();
    void processing();
    
private:
    TxFfeParams m_params;
    std::vector<double> m_buffer;  // 循环缓冲区
    size_t m_buffer_ptr;            // 缓冲区指针
};

} // namespace serdes

#endif // SERDES_TX_FFE_H
