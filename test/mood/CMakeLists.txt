#
# Mood related tests
#

##############################################
add_executable(test_monomials_ordering
  test_monomials_ordering.cpp)
target_include_directories(test_monomials_ordering
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

##############################################
add_executable(test_print_monomial_map
  test_print_monomial_map.cpp)
target_include_directories(test_print_monomial_map
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_print_monomial_map mood kokkos hwloc dl)

##############################################
add_executable(test_polynomial
  test_polynomial.cpp)
target_include_directories(test_polynomial
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_polynomial kokkos hwloc dl)

##############################################
add_executable(test_stencil
  test_stencil.cpp)
target_include_directories(test_stencil
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${OpenBLAS_INCLUDE_DIR}
  )
target_link_libraries(test_stencil mood kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_matrix_qr
  test_matrix_qr.cpp)
target_include_directories(test_matrix_qr
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${OpenBLAS_INCLUDE_DIR}
  )
target_link_libraries(test_matrix_qr mood kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_reconstruct_2d
  test_reconstruct_2d.cpp)
target_include_directories(test_reconstruct_2d
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_reconstruct_2d mood kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_reconstruct_3d
  test_reconstruct_3d.cpp)
target_include_directories(test_reconstruct_3d
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_reconstruct_3d mood kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_mood_functor
  test_mood_functor.cpp)
target_include_directories(test_mood_functor
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_mood_functor shared mood io time config kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_mood_functor_flux
  test_mood_functor_flux.cpp)
target_include_directories(test_mood_functor_flux
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_mood_functor_flux shared mood io time config kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

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


