set (muscl_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/hydro_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDBaseFunctor2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDBaseFunctor3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl2D.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl3D.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMHDMuscl2D.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMHDMuscl2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMHDMuscl3D.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMHDMuscl3D.h
  )

add_library(muscl OBJECT
  ${muscl_SRCS}
  )
target_include_directories(muscl
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ) 
#set_target_properties(muscl PROPERTIES
#  COMPILE_FLAGS "${KOKKOS_CPPFLAGS} ${KOKKOS_CXXFLAGS}")
