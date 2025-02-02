#
# Papi related tests
#

# ##################################################################################################
add_executable(test_papi PapiInfoTest.cpp)
target_include_directories(test_papi PUBLIC ${PAPI_INCLUDE_DIR})
target_link_libraries(test_papi PUBLIC ppkMHD::monitoring kokkos ${PAPI_LIBRARY})
