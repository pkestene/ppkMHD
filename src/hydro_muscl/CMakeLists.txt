set (hydro_muscl_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRun.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRun.h
  ${CMAKE_CURRENT_SOURCE_DIR}/real_type.h
  )

add_library(hydro_muscl OBJECT
  ${hydro_muscl_SRCS}
  )
target_include_directories(hydro_muscl
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/src/utils/time
  ) 
#set_target_properties(hydro_muscl PROPERTIES
#  COMPILE_FLAGS "${KOKKOS_CPPFLAGS} ${KOKKOS_CXXFLAGS}")
