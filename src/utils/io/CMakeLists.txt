set (io_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_Writer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_Writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.cpp
  )

add_library(io OBJECT
  ${io_SRCS}
  )

target_include_directories(io
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/shared
  ${CMAKE_SOURCE_DIR}/src/utils)
