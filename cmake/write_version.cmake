find_package(Git QUIET)
if (NOT GIT_FOUND)
  set(GIT_BUILD_STRING "N/A")
else()
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
    OUTPUT_VARIABLE GIT_BUILD_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
execute_process(COMMAND date "+%d/%m/%y"
  OUTPUT_VARIABLE DATE_STRING
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND date "+%H:%M:%S"
  OUTPUT_VARIABLE TIME_STRING
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(RELEASE_BUILD True)
endif()

configure_file(
  ${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}_version.h.in
  ${PROJECT_BINARY_DIR}/src/${PROJECT_NAME}_version.h)
