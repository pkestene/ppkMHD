add_executable(test_io_vtk
  test_io_vtk.cpp
  )

target_include_directories(test_io_vtk
  PUBLIC
  ${CMAKE_SOURCE_DIR}/kokkos/core/src
  ${CMAKE_SOURCE_DIR}/kokkos/containers/src
  ${CMAKE_SOURCE_DIR}/kokkos/algorithms/src
  ${CMAKE_BINARY_DIR}/kokkos
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils/io
  ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils
  )
target_link_libraries(test_io_vtk
  kokkos shared config monitoring io)

if(USE_MPI)
  target_link_libraries(test_io_vtk
    mpiUtils)
endif(USE_MPI)
