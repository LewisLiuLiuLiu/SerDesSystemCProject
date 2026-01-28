# SystemC-AMS SerDes Modeling
Abstract
This project aims to build a complete SerDes behavioral-level modeling platform based on the EDA-integrated-friendly SystemC-AMS framework, achieving a complete design flow from algorithm verification to system-level performance evaluation, and providing an efficient system architecture design template for high-speed interface design.

## Introduction 
With the continuous increase in data transmission rates of high-speed serial interfaces to 56Gbps and even 112Gbps, SerDes links face increasingly severe signal integrity challenges including channel frequency-dependent attenuation, intersymbol interference, clock jitter, and power supply noise. Traditional SPICE-level transistor simulation methods consume enormous computational resources, making it difficult to support system-level architecture exploration and parameter space search; Python and Matlab system models are very complex to integrate with mainstream EDA software today. This project aims to build a complete SerDes behavioral-level modeling platform based on the EDA-integrated-friendly SystemC-AMS framework, achieving a complete design flow from algorithm verification to system-level performance evaluation, and providing an efficient system architecture design template for high-speed interface design.

## Modeling Detals
This project has established four core technical innovations. 

### SystemC-AMS Multi-Domain Collaborative Mixed-Signal Modeling
First, a SystemC-AMS multi-domain collaborative mixed-signal modeling architecture is established, organically integrating four modeling domains: TDF, LSF, ELN, and DE, and implementing real-time cross-domain parameter transmission through DE-TDF bridging mechanisms. TDF modules declare port rates through the set_attributes() method, and the SystemC-AMS scheduler automatically handles data resampling among multi-rate modules. 

### 1. set_attributes() 方法 - 声明端口速率
文件位置：
- serdes_link_top.h:240 - 顶层模块的set_attributes()
- rx_top.h:176 - RX模块
- tx_driver.h:74 - TX Driver模块
- wave_generation.h:37 - 波形生成模块
- channel_sparam.h:114 - 通道模型
//serdes_link_top.h
void set_attributes() override {
    in.set_rate(1);   // 设置输入端口速率
    out.set_rate(1);  // 设置输出端口速率
}
// serdes_link_top.h 中的信号定义
sca_tdf::sca_signal<double> m_sig_wavegen_out;    // TDF域信号
sca_tdf::sca_signal<double> m_sig_tx_out_p;
sca_tdf::sca_signal<double> m_sig_tx_out_n; 
// adaption_test_common.h 中的控制信号
sc_core::sc_signal<double> sig_phase_error;   // DE域信号（连续时间控制）
sc_core::sc_signal<double> sig_vga_gain;
sc_core::sc_signal<bool> sig_reset;


### Hierarchical Multi-Rate Adaptive Control Architecture
Second, a hierarchical multi-rate adaptive control algorithm system is established, dividing algorithms into two levels: fast path (CDR PI, threshold adaptation, update period approximately 10 to 100 UI) and slow path (AGC, DFE, update period approximately 1000 to 10000 UI), with safety mechanisms including freeze strategy, rollback strategy, rate limitation, saturation clamping, and leakage mechanism. 

//adaption.cpp
void AdaptionDe::fast_path_process() {
    while (true) {
        wait(m_fast_period);  // ~10-100 UI 更新周期
        
        // CDR PI Update
        if (m_params.cdr_pi.enabled) { ... }
        
        // Threshold Adaptation  
        if (m_params.threshold.enabled) { ... }
        
        m_fast_update_count++;
    }
}


### Non-Ideal Effect Comprehensive Physical Modeling
Third, a comprehensive physical modeling method for non-ideal effects is established, systematically modeling random jitter, deterministic jitter, phase noise, input-output offset, output saturation, PSRR, CMRR, and other key effects. The modeling method implements two modes: soft saturation (tanh function) and hard saturation (clamp function) based on physical mechanisms. 

### Software Methodology
Fourth, a configuration-driven extensible system architecture is constructed, defining over 103 system parameters, enabling behavioral adjustment through JSON configuration files without modifying source code. A comprehensive test framework is established with 94 unit tests covering TX, RX, and WaveGen modules, ensuring code reliability and regression prevention. Continuous Integration and Continuous Deployment (CI/CD) pipelines are implemented using GitHub Actions, automating build, test, and verification processes for improved development efficiency and code quality assurance.


## Modeling Results
This project has achieved significant technical breakthroughs. In terms of efficiency, taking a typical scenario with 10Gbps data rate, 80GHz sampling rate, and 1 microsecond simulation duration as an example, complete link simulation can be completed within seconds on a regular PC, achieving an efficiency improvement of 3 to 4 orders of magnitude compared to SPICE-level simulation. Non-ideal effect modeling reduces jitter prediction errors, providing more detailed circuit specifications for subsequent analog circuit design. The project has established a complete test system with 94 unit tests. TX, RX, and WaveGen modules have been successfully reused in multiple projects with a reuse rate of 100%. Both Xrun and VCS can perform direct simulation.

## Conclusions
This project has made four theoretical contributions in the field of SerDes system behavioral-level modeling: established a SystemC-AMS multi-domain collaborative mixed-signal modeling methodology framework, proposed a hierarchical multi-rate adaptive control architecture, established a comprehensive physical modeling methodology for non-ideal effects, and implemented a configuration-driven extensible system architecture. In terms of engineering application value, the balance between simulation speed and accuracy significantly improves design iteration efficiency, and nonlinear physical modeling enables more accurate eye diagram analysis.

Keywords : SerDes; SystemC-AMS; Behavioral-level Modeling; Multi-domain Collaboration; Adaptive Control; Non-ideal Effect Modeling; Eye Diagram Analysis