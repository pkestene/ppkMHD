#
# Papi related tests
#

##############################################
add_executable(test_papi
  PapiInfoTest.cpp)
target_include_directories(test_papi
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${PAPI_INCLUDE_DIR}
  )
target_link_libraries(test_papi monitoring kokkos ${PAPI_LIBRARY})
