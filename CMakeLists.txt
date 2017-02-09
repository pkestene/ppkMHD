cmake_minimum_required(VERSION 2.8.3)
project(ppkMHD CXX)

# Create project version (using git info ?)
# TODO

# #######################################################
# Initialize CMAKE module path (for Kokkos and others)
#########################################################
# Make sure KOKKOS_PATH has been set in environment (e.g. using a modulefile)
if(NOT DEFINED ENV{KOKKOS_PATH})
  message(FATAL_ERROR "KOKKOS_PATH env variable must be set and pointing to Kokkos installation directory, i.e. where cmake can find file kokkos.cmake.")
else()
  # just make sure kokkos install path contains a file named kokkos.cmake
  if (NOT EXISTS $ENV{KOKKOS_PATH}/kokkos.cmake)
    message(FATAL_ERROR "file kokkos.cmake does not exists in \"$ENV{KOKKOS_PATH}\" ! Check your kokkos installation")
  endif()
  set(CMAKE_MODULE_PATH "$ENV{KOKKOS_PATH}" ${CMAKE_MODULE_PATH})
endif()

#
# include Kokkos configuration : kokkos.cmake
#
include(kokkos)

#
# nvcc does like duplicate flags, so we try to filter them out
# http://stackoverflow.com/questions/39758585/duplicate-compile-flag-in-cmake-cxx-flags
#
# USUSED for now.
#
# function(removeDuplicateSubstring stringIn stringOut)
#     separate_arguments(stringIn)
#     list(REMOVE_DUPLICATES stringIn)
#     string(REPLACE ";" " " stringIn "${stringIn}")
#     set(${stringOut} "${stringIn}" PARENT_SCOPE)
# endfunction()


#
# common flags
#
if (KOKKOS_INTERNAL_USE_CUDA MATCHES "1")
  set(CMAKE_CXX_FLAGS  "${KOKKOS_CPPFLAGS} ${KOKKOS_CXXFLAGS} ${CMAKE_CXX_FLAGS} -DUSE_DOUBLE -DCUDA")
else()
  set(CMAKE_CXX_FLAGS  "${KOKKOS_CPPFLAGS} ${KOKKOS_CXXFLAGS} ${CMAKE_CXX_FLAGS} -DUSE_DOUBLE")
endif()
  
set(CMAKE_EXE_LINKER_FLAGS  "${KOKKOS_LDFLAGS} ${CMAKE_EXE_LINKER_FLAGS}")



add_subdirectory(test)
add_subdirectory(src)
