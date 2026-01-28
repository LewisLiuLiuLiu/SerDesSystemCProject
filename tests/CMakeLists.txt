# Tests CMakeLists.txt
# 
# 此文件定义了所有单元测试的构建规则。每个测试都构建为独立的可执行文件，
# 以避免 SystemC 仿真器重置问题。所有测试共享相同的主函数文件 (test_main.cpp)
# 并链接到必要的库。

include(GoogleTest)

# ============================================================================
# 公共函数定义 - 简化测试可执行文件创建过程
# ============================================================================

# 为指定的测试名称列表创建可执行文件的函数
function(create_test_executables test_list)
    foreach(test_name ${test_list})
        add_executable(test_${test_name}
            test_main.cpp
            unit/test_${test_name}.cpp
        )
        
        target_link_libraries(test_${test_name}
            GTest::gtest
            serdes_lib
            ${SYSTEMC_AMS_LIBRARY}
            ${SYSTEMC_LIBRARY}
        )
        
        add_test(NAME ${test_name} COMMAND test_${test_name})
    endforeach()
endfunction()

# ============================================================================
# Adaption 模块测试 - 独立可执行文件
# 测试内容：AGC、CDR PI控制器、DFE自适应算法、阈值自适应、冻结机制等
# ============================================================================

set(ADAPTION_TESTS
    adaption_port_connection        # 端口连接测试
    adaption_param_validation       # 参数验证测试
    adaption_update_count           # 更新计数测试
    adaption_output_range           # 输出范围测试
    adaption_mode_change            # 模式切换测试
    adaption_agc_basic              # AGC基础测试
    adaption_agc_convergence        # AGC收敛测试
    adaption_agc_rate_limit         # AGC速率限制测试
    adaption_dfe_basic              # DFE基础测试
    adaption_dfe_sign_lms           # DFE Sign LMS算法测试
    adaption_dfe_lms                # DFE LMS算法测试
    adaption_cdr_pi_basic           # CDR PI控制器基础测试
    adaption_cdr_pi_antiwindup      # CDR PI防积分饱和测试
    adaption_cdr_pi_convergence     # CDR PI收敛测试
    adaption_threshold_basic        # 阈值基础测试
    adaption_threshold_drift        # 阈值漂移测试
    adaption_threshold_hysteresis   # 阈值迟滞测试
    adaption_freeze_mechanism       # 冻结机制测试
    adaption_rollback               # 回滚机制测试
    adaption_snapshot               # 快照机制测试
)

create_test_executables("${ADAPTION_TESTS}")

# ============================================================================
# CDR 模块测试 - 独立可执行文件（每种测试一个可执行文件）
# 测试内容：PI控制器、PAI、边沿检测、图案识别、参数验证等
# ============================================================================

set(CDR_TESTS
    cdr_basic_functionality         # 基础功能测试
    cdr_debug_interface             # 调试接口测试
    cdr_pi_config                   # PI配置测试
    cdr_pi_proportional             # PI比例项测试
    cdr_pi_integral                 # PI积分项测试
    cdr_pi_configurations           # PI多种配置测试
    cdr_pi_config_high_gain         # PI高增益配置测试
    cdr_pi_config_low_gain          # PI低增益配置测试
    cdr_edge_threshold_config       # 边沿阈值配置测试
    cdr_edge_rising                 # 上升沿检测测试
    cdr_edge_falling                # 下降沿检测测试
    cdr_edge_none                   # 无边沿检测测试
    cdr_edge_threshold_effect       # 边沿阈值效果测试
    cdr_edge_threshold_low          # 低阈值测试
    cdr_validation_kp               # Kp参数验证测试
    cdr_validation_resolution       # 分辨率验证测试
    cdr_validation_range            # 范围验证测试
    cdr_validation_boundary         # 边界验证测试
    cdr_pattern_alternating         # 交替图案测试
    cdr_pattern_low_density         # 低密度图案测试
    cdr_pai_config                  # PAI配置测试
    cdr_pai_range_limit             # PAI范围限制测试
    cdr_pai_quantization            # PAI量化测试
)

create_test_executables("${CDR_TESTS}")

# ============================================================================
# Sampler 模块测试 - 独立可执行文件（每种测试一个可执行文件）
# 测试内容：判决功能、迟滞行为、噪声和偏移、相位采样、输出范围等
# ============================================================================

set(SAMPLER_TESTS
    sampler_basic_decision          # 基础判决测试
    sampler_negative_decision       # 负判决测试
    sampler_hysteresis_behavior     # 迟滞行为测试
    sampler_fuzzy_decision          # 模糊判决测试
    sampler_noise_effect            # 噪声影响测试
    sampler_offset_effect           # 偏移影响测试
    sampler_validation_params       # 参数验证测试
    sampler_validation_phase_source # 相位源验证测试
    sampler_validation_valid_phase_source # 有效相位源验证测试
    sampler_output_range_neg05      # 输出范围负0.5测试
    sampler_output_range_neg01      # 输出范围负0.1测试
    sampler_output_range_zero       # 输出范围零测试
    sampler_output_range_pos01      # 输出范围正0.1测试
    sampler_output_range_pos05      # 输出范围正0.5测试
    sampler_de_output               # DE输出测试
    sampler_de_negative_input       # DE负输入测试
)

create_test_executables("${SAMPLER_TESTS}")

# ============================================================================
# TX FFE 模块测试 - 独立可执行文件（每种测试一个可执行文件）
# 测试内容：多抽头、卷积、频率响应、系数配置、预/去加重等
# ============================================================================

set(FFE_TESTS
    ffe_basic_functionality         # 基础功能测试
    ffe_tap_coefficients            # 抽头系数测试
    ffe_default_params              # 默认参数测试
    ffe_impulse_response            # 脉冲响应测试
    ffe_convolution                 # 卷积测试
    ffe_frequency_response          # 频率响应测试
    ffe_taps_boundary               # 抽头边界测试
    ffe_multi_tap                   # 多抽头测试
    ffe_deemphasis                  # 去加重测试
    ffe_preemphasis                 # 预加重测试
)

create_test_executables("${FFE_TESTS}")

# ============================================================================
# TX Driver 模块测试 - 独立可执行文件（每种测试一个可执行文件）
# 测试内容：DC增益、共模、饱和特性、带宽、PSRR、转换速率等
# ============================================================================

set(TX_DRIVER_TESTS
    tx_driver_dc_gain               # DC增益测试
    tx_driver_common_mode           # 共模测试
    tx_driver_soft_saturation       # 软饱和测试
    tx_driver_hard_saturation       # 硬饱和测试
    tx_driver_bandwidth             # 带宽测试
    tx_driver_psrr_test             # PSRR测试
    tx_driver_gain_mismatch         # 增益失配测试
    tx_driver_slew_rate             # 转换速率测试
)

create_test_executables("${TX_DRIVER_TESTS}")

# ============================================================================
# Wave Generation 模块测试 - 独立可执行文件（每种测试一个可执行文件）
# 测试内容：PRBS序列、NRZ电平、抖动配置、脉冲特性、重现性、稳定性等
# ============================================================================

set(WAVE_GEN_TESTS
    wave_gen_basic_functionality    # 基础功能测试
    wave_gen_nrz_level              # NRZ电平测试
    wave_gen_prbs_mode              # PRBS模式测试
    wave_gen_prbs7                  # PRBS7测试
    wave_gen_prbs9                  # PRBS9测试
    wave_gen_prbs15                 # PRBS15测试
    wave_gen_prbs23                 # PRBS23测试
    wave_gen_prbs31                 # PRBS31测试
    wave_gen_pulse_basic            # 脉冲基础测试
    wave_gen_pulse_timing           # 脉冲时序测试
    wave_gen_invalid_sample_rate    # 无效采样率测试
    wave_gen_invalid_pulse_width    # 无效脉宽测试
    wave_gen_debug_lfsr             # LFSR调试测试
    wave_gen_debug_time             # 时间调试测试
    wave_gen_code_balance           # 码型平衡测试
    wave_gen_mean_value             # 均值测试
    wave_gen_long_stability         # 长期稳定性测试
    wave_gen_jitter_config          # 抖动配置测试
    wave_gen_seed_run1              # 种子运行1测试
    wave_gen_seed_run2              # 种子运行2测试
    wave_gen_repro_run1             # 重现运行1测试
    wave_gen_repro_run2             # 重现运行2测试
)

create_test_executables("${WAVE_GEN_TESTS}")

# ============================================================================
# Clock Generation 模块测试 - 独立可执行文件（每种测试一个可执行文件）
# 测试内容：理想时钟、PLL/ADPLL、不同频率、相位特性、稳定性等
# ============================================================================

set(CLOCK_GEN_TESTS
    clock_gen_ideal_basic           # 理想基础测试
    clock_gen_phase_range           # 相位范围测试
    clock_gen_phase_continuity      # 相位连续性测试
    clock_gen_initial_phase         # 初始相位测试
    clock_gen_freq_1ghz             # 1GHz频率测试
    clock_gen_freq_10ghz            # 10GHz频率测试
    clock_gen_freq_40ghz            # 40GHz频率测试
    clock_gen_freq_80ghz            # 80GHz频率测试
    clock_gen_timestep              # 时间步长测试
    clock_gen_invalid_freq          # 无效频率测试
    clock_gen_extreme_freq          # 极端频率测试
    clock_gen_pll_validation        # PLL验证测试
    clock_gen_cycle_count           # 周期计数测试
    clock_gen_long_stability        # 长期稳定性测试
    clock_gen_mean_phase            # 平均相位测试
    clock_gen_type_ideal            # 理想类型测试
    clock_gen_type_pll              # PLL类型测试
    clock_gen_type_adpll            # ADPLL类型测试
    clock_gen_debug                 # 调试测试
)

create_test_executables("${CLOCK_GEN_TESTS}")

# ============================================================================
# CTLE 和 VGA 测试 - 保持单文件（未拆分）
# 测试内容：端口连接、差分传输、共模输出、DC增益、零点/极点配置等
# ============================================================================

set(CTLE_VGA_TESTS
    ctle_basic                      # CTLE基础测试
    vga_basic                       # VGA基础测试
)

create_test_executables("${CTLE_VGA_TESTS}")

# ============================================================================
# Channel S-Parameter 测试 - 独立可执行文件
# 测试内容：配置加载、VF/IR一致性、卷积处理、DC增益等
# ============================================================================

set(CHANNEL_SPARAM_TESTS
    channel_sparam              # 原有基础测试
    channel_sparam_config       # 配置加载测试
    channel_sparam_processing   # 信号处理测试
)

create_test_executables("${CHANNEL_SPARAM_TESTS}")

# ============================================================================
# TX Top 模块测试 - 独立可执行文件
# 测试内容：基础功能、FFE效果、Driver摆幅、差分信号、饱和特性、带宽、WaveGen集成
# ============================================================================

set(TX_TOP_TESTS
    tx_top_basic                    # 基础功能测试
    tx_top_ffe_effect               # FFE预均衡效果测试
    tx_top_driver_swing             # Driver输出摆幅测试
    tx_top_differential             # 差分信号完整性测试
    tx_top_saturation               # 饱和特性测试
    tx_top_bandwidth                # 带宽限制测试
    tx_top_wavegen_output           # WaveGen集成测试
    tx_top_integration              # 全链路集成测试
)

create_test_executables("${TX_TOP_TESTS}")

# ============================================================================
# 集成测试
# ============================================================================

file(GLOB INTEGRATION_TEST_SOURCES "integration/*.cpp")

if(INTEGRATION_TEST_SOURCES)
    add_executable(integration_tests
        test_main.cpp
        ${INTEGRATION_TEST_SOURCES}
    )
    
    target_link_libraries(integration_tests
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
        GTest::gtest
        GTest::gtest_main
    )
    
    add_test(NAME integration_tests COMMAND integration_tests)
endif()
