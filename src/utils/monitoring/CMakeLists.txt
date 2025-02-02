find_package(OpenMP)

add_library(monitoring "")

target_sources(monitoring PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/OpenMPTimer.cpp
                                  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTimer.cpp)

if(PAPI_FOUND)
  target_sources(monitoring PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/PapiInfo.cpp)
endif(PAPI_FOUND)

target_include_directories(monitoring PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)

if(PAPI_FOUND)
  target_include_directories(monitoring PUBLIC ${PAPI_INCLUDE_DIR})
  target_link_libraries(monitoring PUBLIC ${PAPI_LIBRARY})
endif(PAPI_FOUND)

if(OpenMP_CXX_FOUND)
  target_link_libraries(monitoring PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)

# Add an alias so that library can be used inside the build tree
add_library(ppkMHD::monitoring ALIAS monitoring)
