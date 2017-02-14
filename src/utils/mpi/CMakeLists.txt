set (mpi_env_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_env.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_env.cpp
  )

add_library(mpi_env OBJECT
  ${mpi_env_SRCS}
  )
