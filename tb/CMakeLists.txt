# Testbench CMakeLists.txt

# Simple link testbench
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/simple_link_tb.cpp)
    add_executable(simple_link_tb simple_link_tb.cpp)
    target_link_libraries(simple_link_tb
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
    )
endif()

# Full system testbench
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/full_system_tb.cpp)
    add_executable(full_system_tb full_system_tb.cpp)
    target_link_libraries(full_system_tb
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
    )
endif()

# CTLE transient testbench
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rx/ctle/ctle_tran_tb.cpp)
    add_executable(ctle_tran_tb rx/ctle/ctle_tran_tb.cpp)
    target_include_directories(ctle_tran_tb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rx/ctle
    )
    target_link_libraries(ctle_tran_tb
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
    )
endif()

# VGA transient testbench
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rx/vga/vga_tran_tb.cpp)
    add_executable(vga_tran_tb rx/vga/vga_tran_tb.cpp)
    target_include_directories(vga_tran_tb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rx/vga
    )
    target_link_libraries(vga_tran_tb
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
    )
endif()

# Sampler transient testbench
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rx/sampler/sampler_tran_tb.cpp)
    add_executable(sampler_tran_tb rx/sampler/sampler_tran_tb.cpp)
    target_include_directories(sampler_tran_tb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rx/sampler
    )
    target_link_libraries(sampler_tran_tb
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
    )
endif()

# CDR transient testbench
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rx/cdr/cdr_tran_tb.cpp)
    add_executable(cdr_tran_tb rx/cdr/cdr_tran_tb.cpp)
    target_include_directories(cdr_tran_tb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rx/cdr
    )
    target_link_libraries(cdr_tran_tb
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
    )
endif()

# Wave Generation transient testbench
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/periphery/wave_gen/wave_gen_tran_tb.cpp)
    add_executable(wave_gen_tran_tb periphery/wave_gen/wave_gen_tran_tb.cpp)
    target_include_directories(wave_gen_tran_tb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/periphery/wave_gen
    )
    target_link_libraries(wave_gen_tran_tb
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
    )
endif()

# TX Top transient testbench
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tx/tx_tran_tb.cpp)
    add_executable(tx_tran_tb tx/tx_tran_tb.cpp)
    target_include_directories(tx_tran_tb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tx
    )
    target_link_libraries(tx_tran_tb
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
    )
endif()
