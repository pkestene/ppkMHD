#
# SDM (Spectral Difference Method) related tests
#

##############################################
add_executable(test_sdm_geom "")
target_sources(test_sdm_geom
  PUBLIC
  test_sdm_geom.cpp)
target_link_libraries(test_sdm_geom PUBLIC ppkMHD::sdm kokkos hwloc dl)

##############################################
add_executable(test_sdm_lagrange_sol2flux "")
target_sources(test_sdm_lagrange_sol2flux
  PUBLIC
  test_sdm_lagrange_sol2flux.cpp)
target_link_libraries(test_sdm_lagrange_sol2flux PUBLIC ppkMHD::sdm kokkos hwloc dl)

##############################################
add_executable(test_sdm_lagrange_flux2sol_derivative_interpolation "")
target_sources(test_sdm_lagrange_flux2sol_derivative_interpolation
  PUBLIC
  test_sdm_lagrange_flux2sol_derivative_interpolation.cpp)
target_link_libraries(test_sdm_lagrange_flux2sol_derivative_interpolation PUBLIC ppkMHD::sdm kokkos hwloc dl)

##############################################
add_executable(test_sdm_lagrange_sol2sol_derivative "")
target_sources(test_sdm_lagrange_sol2sol_derivative
  PUBLIC
  test_sdm_lagrange_sol2sol_derivative.cpp)
target_link_libraries(test_sdm_lagrange_sol2sol_derivative PUBLIC ppkMHD::sdm kokkos hwloc dl)

##############################################
add_executable(test_sdm_io "")
target_sources(test_sdm_io
  PUBLIC
  test_sdm_io.cpp)
target_link_libraries(test_sdm_io 
  PUBLIC 
  ppkMHD::sdm
  ppkMHD::config
  ppkMHD::io 
  ppkMHD::shared
  ppkMHD::monitoring
  kokkos
  hwloc
  dl)

if (USE_MPI)
  target_link_libraries(test_sdm_io 
    PUBLIC 
    ppkMHD::mpiUtils)
endif(USE_MPI)


# copy input parameter files for testing SDM IO
configure_file(test_sdm_io_2D.ini test_sdm_io_2D.ini COPYONLY)
configure_file(test_sdm_io_3D.ini test_sdm_io_3D.ini COPYONLY)

##############################################
add_executable(test_sdm_flux "")
target_sources(test_sdm_flux 
  PUBLIC
  test_sdm_flux.cpp)
target_link_libraries(test_sdm_flux 
  PUBLIC 
  ppkMHD::sdm
  ppkMHD::config
  ppkMHD::io
  ppkMHD::shared
  ppkMHD::monitoring
  kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_flux PUBLIC ppkMHD::mpiUtils)
endif(USE_MPI)

##############################################
add_executable(test_sdm_lagrange_functor "")
target_sources(test_sdm_lagrange_functor
  PUBLIC
  test_sdm_lagrange_functor.cpp)
target_link_libraries(test_sdm_lagrange_functor 
  PUBLIC
  ppkMHD::sdm
  ppkMHD::config
  ppkMHD::io
  ppkMHD::shared 
  ppkMHD::monitoring
  kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_lagrange_functor PUBLIC mpiUtils)
endif(USE_MPI)

add_test(NAME sdm_lagrange_functor COMMAND test_sdm_lagrange_functor)

##############################################
add_executable(test_sdm_flux_functor "")
target_sources(test_sdm_flux_functor
  PUBLIC
  test_sdm_flux_functor.cpp)
target_link_libraries(test_sdm_flux_functor
  PUBLIC
  ppkMHD::config
  ppkMHD::io
  ppkMHD::shared
  ppkMHD::monitoring
  kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_flux_functor PUBLIC ppkMHD::mpiUtils)
endif(USE_MPI)

# copy input parameter files for testing SDM IO
configure_file(test_sdm_flux_functor_2D.ini test_sdm_flux_functor_2D.ini COPYONLY)
configure_file(test_sdm_flux_functor_3D.ini test_sdm_flux_functor_3D.ini COPYONLY)

##############################################
add_executable(test_sdm_gradient_velocity "")
target_sources(test_sdm_gradient_velocity
  PUBLIC
  test_sdm_gradient_velocity.cpp)
target_link_libraries(test_sdm_gradient_velocity
  PUBLIC
  ppkMHD::sdm
  ppkMHD::config
  ppkMHD::io
  ppkMHD::shared
  ppkMHD::monitoring 
  kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_gradient_velocity PUBLIC ppkMHD::mpiUtils)
endif(USE_MPI)

# copy input parameter files
configure_file(test_sdm_gradient_velocity_2D.ini test_sdm_gradient_velocity_2D.ini COPYONLY)
configure_file(test_sdm_gradient_velocity_3D.ini test_sdm_gradient_velocity_3D.ini COPYONLY)

# copy a paraview state file just for eye cross-checking the 2d results of gradient
# computations
configure_file(test_sdm_gradient_velocity.pvsm test_sdm_gradient_velocity.pvsm COPYONLY) 
  
##############################################
add_executable(test_sdm_compute_dt_functor "")
target_sources(test_sdm_compute_dt_functor
  PUBLIC
  test_sdm_compute_dt_functor.cpp)
target_link_libraries(test_sdm_compute_dt_functor
  PUBLIC
  ppkMHD::sdm
  ppkMHD::config 
  ppkMHD::io
  ppkMHD::shared
  ppkMHD::monitoring
  kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_compute_dt_functor PUBLIC ppkMHD::mpiUtils)
endif(USE_MPI)

add_test(NAME sdm_compute_dt_functor COMMAND test_sdm_compute_dt_functor)

##############################################
add_executable(test_sdm_chebyshev_quadrature "")
target_sources(test_sdm_chebyshev_quadrature
  PUBLIC
  test_sdm_chebyshev_quadrature.cpp)
target_link_libraries(test_sdm_chebyshev_quadrature 
  PUBLIC
  ppkMHD::sdm
  kokkos hwloc dl)

##############################################
add_executable(test_sdm_isentropic_vortex "")
target_sources(test_sdm_isentropic_vortex
  PUBLIC
  test_sdm_isentropic_vortex.cpp)
target_link_libraries(test_sdm_isentropic_vortex 
  PUBLIC
  ppkMHD::sdm
  ppkMHD::solver_factory 
  ppkMHD::config 
  kokkos
  ppkMHD::muscl 
  ppkMHD::io
  ppkMHD::shared
  ppkMHD::monitoring
  kokkos hwloc dl)
if (USE_MPI)
  target_link_libraries(test_sdm_isentropic_vortex PUBLIC ppkMHD::mpiUtils)
endif(USE_MPI)

##############################################
add_executable(test_sdm_average_functor "")
target_sources(test_sdm_average_functor 
  PUBLIC
  test_sdm_average_functor.cpp)
target_link_libraries(test_sdm_average_functor
  PUBLIC
  ppkMHD::sdm
  ppkMHD::config
  ppkMHD::io
  ppkMHD::shared
  ppkMHD::monitoring
  kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_average_functor PUBLIC ppkMHD::mpiUtils)
endif(USE_MPI)

# copy input parameter files for testing SDM Limiter test
configure_file(test_sdm_limiter_2D.ini test_sdm_limiter_2D.ini COPYONLY)
configure_file(test_sdm_limiter_3D.ini test_sdm_limiter_3D.ini COPYONLY)

##############################################
add_executable(test_sdm_lsq_gradient "")
target_sources(test_sdm_lsq_gradient 
  PUBLIC
  test_sdm_lsq_gradient.cpp)
target_link_libraries(test_sdm_lsq_gradient 
  PUBLIC
  ppkMHD::sdm
  ppkMHD::config
  ppkMHD::io
  ppkMHD::shared 
  ppkMHD::monitoring
  kokkos hwloc dl)

if (USE_MPI)
  target_link_libraries(test_sdm_lsq_gradient PUBLIC ppkMHD::mpiUtils)
endif(USE_MPI)
