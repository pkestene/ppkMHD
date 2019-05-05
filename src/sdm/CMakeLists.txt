# sdm is a header only library
add_library(sdm INTERFACE)

# INTERFACE library requires no source arguments
# target_sources(sdm
#   INTERFACE
#   ${CMAKE_CURRENT_SOURCE_DIR}/sdm_shared.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Geometry.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDMBaseFunctor.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Boundaries_Functors.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Boundaries_Functors_Wedge.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Boundaries_Functors_Jet.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Compute_error.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Dt_Functors.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Flux_Functors.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Flux_with_Limiter_Functors.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Interpolate_Functors.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Limiter_Functors.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Positivity_preserving.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SDM_Run_Functors.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/HydroInitFunctors.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroSDM.h
#   )

target_include_directories(sdm
  INTERFACE
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(sdm
  INTERFACE
  kokkos
  )

add_library(ppkMHD::sdm ALIAS sdm)
