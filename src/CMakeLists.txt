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
add_subdirectory(mood)

# include header for main application
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# ppkMHD executable
#
set(EXE_NAME "${PROJECT_NAME}")
add_executable(${EXE_NAME}
  main.cpp
  )

target_include_directories(${EXE_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

target_link_libraries(${EXE_NAME} solver_factory shared mood muscl io time config kokkos hwloc dl ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})
