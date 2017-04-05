#######################
### SHARED ############
#######################
set (shared_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/BlastParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/KHParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/WedgeParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/initRiemannConfig2d.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundariesFunctors.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundariesFunctorsWedge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroState.h
  ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/real_type.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSolvers.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  )

add_library(shared OBJECT
  ${shared_SRCS}
  )
target_include_directories(shared
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/src/utils/time
  ${CMAKE_SOURCE_DIR}/src/utils/io
  ) 

#######################
### SOLVER FACTORY ####
#######################
set (solver_factory_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverFactory.h
  )
add_library(solver_factory OBJECT
  ${solver_factory_SRCS}
  )
target_include_directories(solver_factory
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/muscl
)
