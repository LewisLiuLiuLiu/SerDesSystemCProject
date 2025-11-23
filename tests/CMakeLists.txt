# Tests CMakeLists.txt

include(GoogleTest)

# Collect all test source files
file(GLOB UNIT_TEST_SOURCES "unit/*.cpp")
file(GLOB INTEGRATION_TEST_SOURCES "integration/*.cpp")

set(ALL_TEST_SOURCES
    ${UNIT_TEST_SOURCES}
    ${INTEGRATION_TEST_SOURCES}
)

# Unit tests executable
if(UNIT_TEST_SOURCES)
    add_executable(unit_tests
        test_main.cpp
        ${UNIT_TEST_SOURCES}
    )
    
    target_link_libraries(unit_tests
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
        GTest::gtest
        GTest::gtest_main
    )
    
    gtest_discover_tests(unit_tests)
endif()

# Integration tests executable
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
    
    gtest_discover_tests(integration_tests)
endif()

# All tests executable
if(ALL_TEST_SOURCES)
    add_executable(all_tests
        test_main.cpp
        ${ALL_TEST_SOURCES}
    )
    
    target_link_libraries(all_tests
        serdes_lib
        ${SYSTEMC_AMS_LIBRARY}
        ${SYSTEMC_LIBRARY}
        GTest::gtest
        GTest::gtest_main
    )
    
    gtest_discover_tests(all_tests)
endif()
