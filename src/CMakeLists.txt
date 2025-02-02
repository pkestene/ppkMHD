# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)

# common solver code
add_subdirectory(shared)

# Hydro solver using 2nd order Muscl-Hancock solver
add_subdirectory(muscl)

# Hydro solver using high order MOOD solver
if(PPKMHD_USE_MOOD)
  add_subdirectory(mood)
endif()

# Hydro solver using high order MOOD solver
if(PPKMHD_USE_SDM)
  add_subdirectory(sdm)
endif()

# include header for main application
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(PAPI_FOUND)
  include_directories(${PAPI_INCLUDE_DIR})
endif()

#
# ppkMHD executable
#
add_executable(${PROJECT_NAME} "")

target_sources(${PROJECT_NAME} PRIVATE main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${CMAKE_BINARY_DIR}/src)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC ppkMHD::solver_factory
         ppkMHD::config
         kokkos
         dl
         ppkMHD::muscl
         ppkMHD::shared
         ppkMHD::io
         ppkMHD::monitoring)

if(PPKMHD_USE_SDM)
  target_link_libraries(${PROJECT_NAME} PUBLIC ppkMHD::sdm)
endif()

if(PPKMHD_USE_MOOD)
  target_link_libraries(${PROJECT_NAME} PUBLIC ppkMHD::mood ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})
endif()

if(PPKMHD_USE_MPI)
  target_link_libraries(${PROJECT_NAME} PUBLIC ppkMHD::mpiUtils)
endif()
