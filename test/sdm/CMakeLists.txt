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

##############################################
add_executable(test_sdm_lagrange
  test_sdm_lagrange.cpp)
target_include_directories(test_sdm_lagrange
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_lagrange kokkos hwloc dl)

##############################################
add_executable(test_sdm_lagrange_derivative
  test_sdm_lagrange_derivative.cpp)
target_include_directories(test_sdm_lagrange_derivative
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_lagrange_derivative kokkos hwloc dl)

##############################################
add_executable(test_sdm_io
  test_sdm_io.cpp)
target_include_directories(test_sdm_io
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_io config io shared monitoring kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_io mpiUtils)
endif()


# copy input parameter files for testing SDM IO
configure_file(test_sdm_io_2D.ini test_sdm_io_2D.ini COPYONLY)
configure_file(test_sdm_io_3D.ini test_sdm_io_3D.ini COPYONLY)

##############################################
add_executable(test_sdm_flux
  test_sdm_flux.cpp)
target_include_directories(test_sdm_flux
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_flux config io shared monitoring kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_flux mpiUtils)
endif()

##############################################
add_executable(test_sdm_lagrange_functor
  test_sdm_lagrange_functor.cpp)
target_include_directories(test_sdm_lagrange_functor
  PUBLIC
  .
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_lagrange_functor config io shared monitoring kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_lagrange_functor mpiUtils)
endif()

##############################################
add_executable(test_sdm_flux_functor
  test_sdm_flux_functor.cpp)
target_include_directories(test_sdm_flux_functor
  PUBLIC
  .
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_flux_functor config io shared monitoring kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_flux_functor mpiUtils)
endif()
# copy input parameter files for testing SDM IO
configure_file(test_sdm_flux_functor_2D.ini test_sdm_flux_functor_2D.ini COPYONLY)
configure_file(test_sdm_flux_functor_3D.ini test_sdm_flux_functor_3D.ini COPYONLY)

##############################################
add_executable(test_sdm_compute_dt_functor
  test_sdm_compute_dt_functor.cpp)
target_include_directories(test_sdm_compute_dt_functor
  PUBLIC
  .
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_compute_dt_functor config io shared monitoring kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_compute_functor mpiUtils)
endif()

##############################################
add_executable(test_sdm_chebyshev_quadrature
  test_sdm_chebyshev_quadrature.cpp)
target_include_directories(test_sdm_chebyshev_quadrature
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_sdm_chebyshev_quadrature kokkos hwloc dl)
