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
  ${CMAKE_SOURCE_DIR}/src/mood/polynomials_utils.h
  ${CMAKE_SOURCE_DIR}/src/mood/polynomials_utils.cpp
  test_print_monomial_map.cpp)
target_include_directories(test_print_monomial_map
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_print_monomial_map kokkos hwloc dl)

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
  $<TARGET_OBJECTS:mood>
  test_stencil.cpp)
target_include_directories(test_stencil
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${OpenBLAS_INCLUDE_DIR}
  )
#set_target_properties(test_stencil PROPERTIES
#  LINK_FLAGS "${KOKKOS_LDFLAGS}")
#target_link_libraries(test_stencil kokkos hwloc dl ${LAPACKE_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(test_stencil kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_matrix_qr
  $<TARGET_OBJECTS:mood>
  test_matrix_qr.cpp)
target_include_directories(test_matrix_qr
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${OpenBLAS_INCLUDE_DIR}
  )
#set_target_properties(test_matrix_qr PROPERTIES
#  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(test_matrix_qr kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_reconstruct_2d
  $<TARGET_OBJECTS:mood>
  test_reconstruct_2d.cpp)
target_include_directories(test_reconstruct_2d
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
#set_target_properties(test_reconstruct PROPERTIES
#  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(test_reconstruct_2d kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_reconstruct_3d
  $<TARGET_OBJECTS:mood>
  test_reconstruct_3d.cpp)
target_include_directories(test_reconstruct_3d
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
#set_target_properties(test_reconstruct PROPERTIES
#  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(test_reconstruct_3d kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_mood_functor
  $<TARGET_OBJECTS:mood>
  $<TARGET_OBJECTS:config>
  $<TARGET_OBJECTS:shared>
  $<TARGET_OBJECTS:io>
  $<TARGET_OBJECTS:monitoring>
  test_mood_functor.cpp)
target_include_directories(test_mood_functor
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
#set_target_properties(test_reconstruct PROPERTIES
#  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(test_mood_functor kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

##############################################
add_executable(test_mood_functor_flux
  $<TARGET_OBJECTS:mood>
  $<TARGET_OBJECTS:config>
  $<TARGET_OBJECTS:shared>
  $<TARGET_OBJECTS:io>
  $<TARGET_OBJECTS:monitoring>
  test_mood_functor_flux.cpp)
target_include_directories(test_mood_functor_flux
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
#set_target_properties(test_reconstruct PROPERTIES
#  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(test_mood_functor_flux kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})

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


