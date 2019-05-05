# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)

# common solver code
add_subdirectory(shared)

# Hydro solver using 2nd order Muscl-Hancock solver
add_subdirectory(muscl)

# Hydro solver using high order MOOD solver
if(USE_MOOD)
  add_subdirectory(mood)
endif(USE_MOOD)

# Hydro solver using high order MOOD solver
if(USE_SDM)
  add_subdirectory(sdm)
endif(USE_SDM)

# include header for main application
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if (PAPI_FOUND)
  include_directories(${PAPI_INCLUDE_DIR})
endif(PAPI_FOUND)

#
# ppkMHD executable
#
add_executable(${PROJECT_NAME} "")

target_sources(${PROJECT_NAME}
  PRIVATE
  main.cpp
  #${BACKWARD_ENABLE}
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/src
  )

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  ppkMHD::solver_factory
  ppkMHD::config
  kokkos
  dl
  ppkMHD::muscl
  ppkMHD::shared
  ppkMHD::io
  ppkMHD::monitoring)

#add_backward(${PROJECT_NAME})

if (USE_SDM)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ppkMHD::sdm)
endif()

if (USE_MOOD)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ppkMHD::mood 
    ${LAPACKE_LIBRARIES}
    ${OpenBLAS_LIB})
endif(USE_MOOD)

if (USE_MPI)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ppkMHD::mpiUtils)
endif(USE_MPI)
