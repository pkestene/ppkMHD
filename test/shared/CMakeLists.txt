#
# SHARED related tests (src/shared)
#

##############################################
add_executable(test_euler_eigen_decomposition
  test_euler_eigen_decomposition.cpp)
target_include_directories(test_euler_eigen_decomposition
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_euler_eigen_decomposition kokkos dl)

