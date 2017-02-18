#
# Kokkos test example : query device information
#

# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)

# Hydro solver using 2nd order Muscl-Hancock solver
add_subdirectory(muscl)

# include header for main application
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/muscl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/config)

#
# ppkMHD executable
#
set(EXE_NAME "${PROJECT_NAME}")
add_executable(${EXE_NAME}
  main.cpp
  $<TARGET_OBJECTS:muscl>
  $<TARGET_OBJECTS:config>
  $<TARGET_OBJECTS:time>
  )

# set_target_properties(${EXE_NAME} PROPERTIES
#   COMPILE_FLAGS "${KOKKOS_CPPFLAGS} ${KOKKOS_CXXFLAGS}")
target_include_directories(${EXE_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/muscl
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/time
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/config
  )

set_target_properties(${EXE_NAME} PROPERTIES
  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(${EXE_NAME} kokkos hwloc dl)
