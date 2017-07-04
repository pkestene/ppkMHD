set (sdm_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/sdm_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Geometry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SDMBaseFunctor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroInitFunctors.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroSDM.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Run_Functors.h
  )

add_library(sdm
  ${sdm_SRCS}
  )
target_include_directories(sdm
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/kokkos/core/src
  ${CMAKE_SOURCE_DIR}/kokkos/containers/src
  ${CMAKE_SOURCE_DIR}/kokkos/algorithms/src
  ${CMAKE_BINARY_DIR}/kokkos
  ) 
target_link_libraries(sdm
  kokkos
  )
