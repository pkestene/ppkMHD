#
# SDM (Spectral Difference Method) related tests
#

##############################################
add_executable(test_sdm_geom
  test_sdm_geom.cpp)
target_include_directories(test_sdm_geom
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_geom kokkos hwloc dl)

