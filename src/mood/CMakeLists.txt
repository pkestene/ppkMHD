set (mood_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/monomials_ordering.h
  ${CMAKE_CURRENT_SOURCE_DIR}/monomials_print_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Polynomial.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Polynomial.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/polynomials_utils.h
  )

add_library(mood OBJECT
  ${mood_SRCS}
  )
target_include_directories(mood
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
