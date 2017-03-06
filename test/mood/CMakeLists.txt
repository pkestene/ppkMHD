#
# Mood related tests
#

add_executable(test_monomials_ordering
  test_monomials_ordering.cpp)
target_include_directories(test_monomials_ordering
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )


