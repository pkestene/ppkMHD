#
# Kokkos test example : query device information
#

# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)

# common solver code
add_subdirectory(shared)

# Hydro solver using 2nd order Muscl-Hancock solver
add_subdirectory(muscl)

# Hydro solver using high order MOOD solver
if(USE_MOOD)
  add_subdirectory(mood)
endif()

# include header for main application
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if (PAPI_FOUND)
  include_directories(${PAPI_INCLUDE_DIR})
endif()

#
# ppkMHD executable
#
set(EXE_NAME "${PROJECT_NAME}")
add_executable(${EXE_NAME}
  main.cpp
  ${BACKWARD_ENABLE}
  )

target_include_directories(${EXE_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

#target_link_libraries(${EXE_NAME} solver_factory config kokkos hwloc dl muscl io shared monitoring)
target_link_libraries(${EXE_NAME} solver_factory config kokkos dl muscl shared io monitoring)

add_backward(${EXE_NAME})

# if (USE_SDM)
#   target_link_libraries(${EXE_NAME} sdm)
# endif()

if (USE_MOOD)
  target_link_libraries(${EXE_NAME} mood ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})
endif()

if (USE_MPI)
  target_link_libraries(${EXE_NAME} mpiUtils)
endif()
