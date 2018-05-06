#
# ConfigMap class test (src/util/config)
#

##############################################
add_executable(ConfigMapTest
  ConfigMapTest.cpp)
target_include_directories(ConfigMapTest
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils/config
  )
target_link_libraries(ConfigMapTest config)

