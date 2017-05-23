set (io_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_Writer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_Writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.cpp
  )

if(USE_HDF5)
  set (io_SRCS
    ${io_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/IO_HDF5.h
    ${CMAKE_CURRENT_SOURCE_DIR}/IO_HDF5.cpp
    )
endif(USE_HDF5)

add_library(io
  ${io_SRCS}
  )

target_include_directories(io
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(io
  kokkos
  )
