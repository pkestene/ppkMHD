#
# Mood related tests
#

##############################################
add_executable(test_monomials_ordering "")
target_sources(test_monomials_ordering
  PUBLIC
  test_monomials_ordering.cpp)
target_link_libraries(test_monomials_ordering
  PUBLIC
  ppkMHD::mood )

##############################################
add_executable(test_print_monomial_map "")
target_sources(test_print_monomial_map
  PUBLIC
  test_print_monomial_map.cpp)
target_link_libraries(test_print_monomial_map
  PUBLIC
  ppkMHD::mood
  kokkos hwloc dl)

##############################################
add_executable(test_polynomial "")
target_sources(test_polynomial
  PUBLIC
  test_polynomial.cpp)
target_link_libraries(test_polynomial
  PUBLIC
  ppkMHD::mood
  kokkos hwloc dl)

##############################################
add_executable(test_stencil "")
target_sources(test_stencil
  PUBLIC
  test_stencil.cpp)
target_link_libraries(test_stencil
  PUBLIC
  ppkMHD::mood
  kokkos hwloc dl)

##############################################
add_executable(test_matrix_qr "")
target_sources(test_matrix_qr
  PUBLIC
  test_matrix_qr.cpp)
target_link_libraries(test_matrix_qr
  PUBLIC
  ppkMHD::mood
  kokkos hwloc dl)

##############################################
add_executable(test_reconstruct_2d "")
target_sources(test_reconstruct_2d
  PUBLIC
  test_reconstruct_2d.cpp)
target_link_libraries(test_reconstruct_2d
  PUBLIC
  ppkMHD::mood
  kokkos hwloc dl)

##############################################
add_executable(test_reconstruct_3d "")
target_sources(test_reconstruct_3d
  PUBLIC
  test_reconstruct_3d.cpp)
target_link_libraries(test_reconstruct_3d
  PUBLIC
  ppkMHD::mood
  kokkos hwloc dl)

##############################################
add_executable(test_mood_functor "")
target_sources(test_mood_functor
  PUBLIC
  test_mood_functor.cpp)
target_link_libraries(test_mood_functor
  PUBLIC
  ppkMHD::shared
  ppkMHD::mood
  ppkMHD::io
  ppkMHD::monitoring
  ppkMHD::config
  kokkos hwloc dl
  ${PAPI_LIBRARY})
if (PPKMHD_USE_MPI)
  target_link_libraries(test_mood_functor PUBLIC ppkMHD::mpiUtils)
endif(PPKMHD_USE_MPI)

##############################################
add_executable(test_mood_functor_flux "")
target_sources(test_mood_functor_flux
  PUBLIC
  test_mood_functor_flux.cpp)
target_link_libraries(test_mood_functor_flux
  PUBLIC
  ppkMHD::shared
  ppkMHD::mood
  ppkMHD::io
  ppkMHD::monitoring
  ppkMHD::config
  kokkos hwloc dl
  ${PAPI_LIBRARY})
if (PPKMHD_USE_MPI)
  target_link_libraries(test_mood_functor_flux PUBLIC ppkMHD::mpiUtils)
endif(PPKMHD_USE_MPI)

##############################################
# add_executable(test_kokkos_eigen
#   test_kokkos_eigen.cpp)
# target_include_directories(test_kokkos_eigen
#   PUBLIC
#   ${CMAKE_SOURCE_DIR}/src
#   ${EIGEN3_INCLUDE_DIRS}
#   )
# #set_target_properties(test_kokkos_eigen PROPERTIES
# #  LINK_FLAGS "${KOKKOS_LDFLAGS}")
# target_link_libraries(test_kokkos_eigen kokkos hwloc dl)
