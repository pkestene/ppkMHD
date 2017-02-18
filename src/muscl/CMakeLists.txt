set (muscl_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/real_type.h
  ${CMAKE_CURRENT_SOURCE_DIR}/hydro_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMuscl2D.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMuscl2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMuscl3D.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMuscl3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverFactory.h
  )

add_library(muscl OBJECT
  ${muscl_SRCS}
  )
target_include_directories(muscl
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/src/utils/time
  ) 
#set_target_properties(muscl PROPERTIES
#  COMPILE_FLAGS "${KOKKOS_CPPFLAGS} ${KOKKOS_CXXFLAGS}")
