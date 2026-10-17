#include "ams/data_strobe_latch.h"

namespace serdes {

DataStrobeLatchTdf::DataStrobeLatchTdf(sc_core::sc_module_name nm)
    : sca_tdf::sca_module(nm)
    , in("in")
    , data_strobe("data_strobe")
    , out("out")
    , m_held(0.0)
{
}

void DataStrobeLatchTdf::set_attributes() {
    in.set_rate(1);
    in.set_delay(1);
    data_strobe.set_rate(1);
    data_strobe.set_delay(1);   // 与 CDR 输出延迟叠加共 2 步，对齐数据判决
    out.set_rate(1);
    // 继承所在 TDF cluster 的时间步
}

void DataStrobeLatchTdf::initialize() {
    m_held = 0.0;
}

void DataStrobeLatchTdf::processing() {
    if (data_strobe.read()) {
        m_held = in.read();
    }
    out.write(m_held);
}

} // namespace serdes
