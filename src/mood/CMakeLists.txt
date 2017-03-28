set (mood_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/monomials_ordering.h
  ${CMAKE_CURRENT_SOURCE_DIR}/monomials_print_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Polynomial.h
  ${CMAKE_CURRENT_SOURCE_DIR}/polynomials_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/polynomials_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MonomialMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Stencil.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Stencil.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/StencilUtils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/StencilUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/GeometricTerms.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GeometricTerms.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Matrix.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mood_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mood_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MoodBaseFunctor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MoodBaseQuad.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MoodBaseQuad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MoodFunctors.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MoodInitFunctors.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMood.h
  )

add_library(mood OBJECT
  ${mood_SRCS}
  )
target_include_directories(mood
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
