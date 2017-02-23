set (io_writers_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_Writer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_Writer.cpp
  )

add_library(io_writers OBJECT
  ${io_writers_SRCS}
  )

target_include_directories(io_writers
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/muscl
  ${CMAKE_SOURCE_DIR}/src/utils)
