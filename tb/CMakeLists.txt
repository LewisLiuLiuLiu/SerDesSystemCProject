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
