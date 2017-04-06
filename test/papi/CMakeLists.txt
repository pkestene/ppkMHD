#
# Papi related tests
#

##############################################
add_executable(test_papi
  $<TARGET_OBJECTS:monitoring>
  PapiInfoTest.cpp)
target_include_directories(test_papi
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${PAPI_INCLUDE_DIR}
  )
target_link_libraries(test_papi ${PAPI_LIBRARY})
