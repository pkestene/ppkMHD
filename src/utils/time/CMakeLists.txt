set (time_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Timer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaTimer.h
  )

add_library(time OBJECT
  ${time_SRCS}
  )
