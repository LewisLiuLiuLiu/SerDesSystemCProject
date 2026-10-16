/**
 * @file test_cdr_trigger_timebase.cpp
 * @brief RxCdrTdf 采样触发时基正确性回归（P0-1 / F-22）
 *
 * 时基语义（2026-10 整数化重写，见 rx_cdr.h）：
 *   edge 触发:  k*ui + qq
 *   data 触发:  k*ui + sample_point*ui + qq   (sample_point 默认 0.5)
 *   qq = 量化后的 PI 相位修正；时刻按 ceil 对齐仿真步格。
 *
 * 回归目标（旧实现的失效形态）：
 *   - 任意 timestep 下每 UI 恰好 2 个触发（旧实现 ts=ui/10 时 100 UI
 *     仅 63 个，出现 16 UI 的采样空档；ts=ui/7 时丢近半且间距紊乱）
 *   - 触发间距在步格量化界内保持 ui/2
 *   - BBPD 相位误差经 phase_error_de 正确导出（|err| ≤ ui，非零）
 *
 * 双 bench 同进程（不同 timestep 的两个 TDF cluster），单次 sc_start。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/rx_cdr.h"
#include <cmath>
#include <vector>

using namespace serdes;

namespace {

// 每 UI 交替 ±1 的差分源（真实数据速率；PI 随跳变脉冲活动）
class AltDiffSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    AltDiffSource(sc_core::sc_module_name nm, double ts, double ui)
        : sca_tdf::sca_module(nm), out("out"), m_ts(ts), m_ui(ui), m_k(0) {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(m_ts, sc_core::SC_SEC);
    }
    void processing() override {
        // 每个 UI 翻转一次（真实数据速率，而非每步翻转）
        long long ui_idx =
            static_cast<long long>(get_time().to_seconds() / m_ui + 1e-9);
        out.write((ui_idx % 2 == 0) ? 1.0 : -1.0);
        ++m_k;
    }
private:
    double m_ts;
    double m_ui;
    long long m_k;
};

// 触发记录器
class TriggerRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<bool> in;
    std::vector<double> fire_times;
    TriggerRecorder(sc_core::sc_module_name nm, double ts)
        : sca_tdf::sca_module(nm), in("in"), m_ts(ts) {}
    void set_attributes() override {
        in.set_rate(1);
        in.set_timestep(m_ts, sc_core::SC_SEC);
    }
    void processing() override {
        if (in.read()) fire_times.push_back(get_time().to_seconds());
    }
    double max_gap() const {
        double g = 0.0;
        for (size_t i = 1; i < fire_times.size(); ++i) {
            g = std::max(g, fire_times[i] - fire_times[i-1]);
        }
        return g;
    }
    double min_gap() const {
        double g = 1e9;
        for (size_t i = 1; i < fire_times.size(); ++i) {
            g = std::min(g, fire_times[i] - fire_times[i-1]);
        }
        return g;
    }
private:
    double m_ts;
};

// phase_error_de 事件监视器（SC_THREAD 敏感于 DE 信号，统计每次写入）
class PeMonitor : public sc_core::sc_module {
public:
    sc_core::sc_in<double> in;
    int n_nonzero = 0;
    int n_total = 0;
    double max_abs = 0.0;
    bool all_valid = true;

    SC_HAS_PROCESS(PeMonitor);
    PeMonitor(sc_core::sc_module_name nm, double ui)
        : sc_core::sc_module(nm), in("in"), m_ui(ui) {
        SC_THREAD(run);
        sensitive << in;
    }
    void run() {
        while (true) {
            wait();
            double v = in.read();
            ++n_total;
            if (v != 0.0) {
                ++n_nonzero;
                max_abs = std::max(max_abs, std::abs(v));
            }
            // 合法值仅 {0, ±ui}
            if (!(v == 0.0 || std::abs(std::abs(v) - m_ui) < m_ui * 1e-9)) {
                all_valid = false;
            }
        }
    }
private:
    double m_ui;
};

struct CdrTimebaseBench {
    AltDiffSource* src;
    RxCdrTdf* cdr;
    TriggerRecorder* rec;
    sca_tdf::sca_signal<double> s_in;
    sca_tdf::sca_signal<double> s_phase;
    sca_tdf::sca_signal<bool> s_trig;
    sc_core::sc_signal<double> s_pe_de;
    PeMonitor* pe_mon;

    CdrTimebaseBench(const char* pfx, double ts, double ui)
    {
        CdrParams p;
        p.ui = ui;
        p.pi.kp = 0.01;
        p.pi.ki = 1e-4;
        p.pi.edge_threshold = 0.5;
        p.pai.resolution = 1e-12;
        p.pai.range = 5e-11;
        src = new AltDiffSource((std::string(pfx) + "_src").c_str(), ts, ui);
        cdr = new RxCdrTdf((std::string(pfx) + "_cdr").c_str(), p);
        rec = new TriggerRecorder((std::string(pfx) + "_rec").c_str(), ts);
        pe_mon = new PeMonitor((std::string(pfx) + "_pemon").c_str(), ui);
        src->out(s_in);
        cdr->in(s_in);
        cdr->phase_out(s_phase);
        cdr->sampling_trigger(s_trig);
        cdr->phase_error_de(s_pe_de);
        rec->in(s_trig);
        pe_mon->in(s_pe_de);
    }
};

// 断言辅助：单 bench 的时基检查
void check_timebase(const CdrTimebaseBench& b, const char* name,
                    double ui, double ts, double sim_time)
{
    const double n_ui = sim_time / ui;

    // 1) 计数：每 UI 恰好 2 个触发（±3 允许首/尾边界）
    const size_t count = b.rec->fire_times.size();
    EXPECT_NEAR(static_cast<double>(count), 2.0 * n_ui, 3.0)
        << name << ": trigger count must be 2 per UI";

    // 2) 无采样空档：任意相邻触发间隔 < ui + 2步（旧实现出现 16 UI 空档）
    EXPECT_LT(b.rec->max_gap(), ui + 2.0 * ts)
        << name << ": no gap larger than one UI (+ step-grid slack)";

    // 3) 间距规则：所有间隔落在 [ui/2 - ts, ui/2 + 2ts]（ceil 量化界内）
    EXPECT_GT(b.rec->min_gap(), ui / 2.0 - 2.0 * ts)
        << name << ": gaps must not be much shorter than ui/2";
    EXPECT_LT(b.rec->max_gap(), ui / 2.0 + 2.0 * ts)
        << name << ": gaps must not exceed ui/2 by more than step quantization";

    // 4) phase_error_de 导出：事件驱动写变更；值为 {0, ±ui}；有足够多的非零误差
    EXPECT_TRUE(b.pe_mon->all_valid)
        << name << ": phase_error_de values must be in {0, ±ui}";
    EXPECT_GT(b.pe_mon->n_nonzero, 10)
        << name << ": alternating pattern must produce nonzero BBPD errors";
    EXPECT_LE(b.pe_mon->max_abs, ui * (1.0 + 1e-9))
        << name << ": |phase error| must not exceed one UI";
    EXPECT_GT(b.pe_mon->n_total, 10)
        << name << ": phase_error_de must actually be written (event-driven)";
}

} // namespace

TEST(CdrTriggerTimebaseTest, ExactTwoTriggersPerUi) {
    const double ui = 100e-12;
    const double sim_time = 10e-9;  // 100 UI

    // bench A: ts = ui/10（整除）
    CdrTimebaseBench a("a", ui / 10.0, ui);
    // bench B: ts = ui/7（不整除，ceil 量化路径）
    CdrTimebaseBench b("b", ui / 7.0, ui);

    sc_core::sc_start(sim_time, sc_core::SC_SEC);

    check_timebase(a, "A(ts=ui/10)", ui, ui / 10.0, sim_time);
    check_timebase(b, "B(ts=ui/7)", ui, ui / 7.0, sim_time);

    sc_core::sc_stop();
}
