#
# Mood related tests
#

add_executable(test_monomials_ordering
  test_monomials_ordering.cpp)
target_include_directories(test_monomials_ordering
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

add_executable(test_print_monomial_map
  test_print_monomial_map.cpp)
target_include_directories(test_print_monomial_map
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

add_executable(test_polynomial
  test_polynomial.cpp)
target_include_directories(test_polynomial
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

add_executable(test_stencil
  $<TARGET_OBJECTS:mood>
  test_stencil.cpp)
target_include_directories(test_stencil
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
#set_target_properties(test_stencil PROPERTIES
#  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(test_stencil kokkos hwloc dl)


