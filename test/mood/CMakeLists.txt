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

##############################################
add_executable(test_polynomial
  test_polynomial.cpp)
target_include_directories(test_polynomial
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

##############################################
add_executable(test_stencil
  $<TARGET_OBJECTS:mood>
  test_stencil.cpp)
target_include_directories(test_stencil
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
#set_target_properties(test_stencil PROPERTIES
#  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(test_stencil kokkos hwloc dl ${LAPACKE_LIBRARIES} ${BLAS_LIBRARIES})

##############################################
# add_executable(test_reconstruct
#   $<TARGET_OBJECTS:mood>
#   test_reconstruct.cpp)
# target_include_directories(test_reconstruct
#   PUBLIC
#   ${CMAKE_SOURCE_DIR}/src
#   ${EIGEN3_INCLUDE_DIRS}
#   )
# #set_target_properties(test_reconstruct PROPERTIES
# #  LINK_FLAGS "${KOKKOS_LDFLAGS}")
# target_link_libraries(test_reconstruct kokkos hwloc dl)

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


