# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.18)

#
# default local cmake macro repository
#
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#
# Prevent from build in source tree
#
include(preventBuildInSource)

#
# Create project version (using git info ?)
# TODO
#

#
# Init build type: Release, Debug, ...
#
include(initBuildType)

# options
option (PPKMHD_BUILD_DOC  "Enable / disable documentation build (sphinx/html)" OFF)
option (PPKMHD_USE_SPHINX_EXHALE "Enable / disable building API documentation (very long)" OFF)
option (PPKMHD_USE_MPI "Activate / want MPI build" OFF)
option (PPKMHD_USE_VTK "Activate / want VTK build" OFF)
option (PPKMHD_USE_DOUBLE "build with double precision" ON)
option (PPKMHD_USE_MOOD "build MOOD numerical schemes" OFF)
option (PPKMHD_USE_SDM "build Spectral Difference Method numerical schemes" OFF)
option (PPKMHD_USE_HDF5 "build HDF5 input/output support" OFF)
option (PPKMHD_USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option (PPKMHD_USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)
option (PPKMHD_USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)

# disable base languages
unset(PROJECT_LANGUAGES)

set(PROJECT_LANGUAGES ${PROJECT_LANGUAGES} C CXX)

project(ppkMHD
  LANGUAGES ${PROJECT_LANGUAGES})

# Documentation type
if(PPKMHD_BUILD_DOC)
  set(PPKMHD_DOC_TYPE "Undefined" CACHE STRING
    "The documentation type to generate. Available values are html and doxygen")
  # Set the possible values for documentation type
  set_property(CACHE PPKMHD_DOC_TYPE PROPERTY STRINGS "html" "doxygen" "Undefined")
  add_subdirectory(doc)
  return()
endif()

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++17 is for Kokkos
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# Write a small header with build data, git version, etc...
#
include(write_version)

#####################################################################
# External packages: MPI, ...
#####################################################################

#####################################################################
#find_package(MPI REQUIRED)
find_package(MPI)
if (PPKMHD_USE_MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})

    #set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)

    # Full command line to probe if cuda support in MPI implementation is enabled
    # ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
    if (OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO}
        OUTPUT_VARIABLE _output)
      if ( (_output MATCHES "smcuda") OR (PPKMHD_USE_MPI_CUDA_AWARE_ENFORCED) )
        set(MPI_CUDA_AWARE_ENABLED True)
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        set(MPI_CUDA_AWARE_ENABLED False)
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
	add_compile_options(-DMPI_CUDA_AWARE_OFF)
      endif()
    endif()
  else()
    message(WARNING "Not compiling with MPI. Suppress this warning with -DPPKMHD_USE_MPI=OFF")
    set(PPKMHD_USE_MPI OFF)
  endif()
endif()

if (PPKMHD_USE_VTK)
  # look for VTK only if requested; VTK macro might even be not present
  # on the target platform
  find_package(VTK COMPONENTS
    CommonCore
    CommonDataModel
    CommonExecutionModel
    CommonMath
    CommonMisc
    CommonSystem
    CommonTransforms
    IOCore
    IOGeometry
    IOImage
    IOLegacy
    IOXML
    IOXMLParser
    ParallelCore
    ParallelMPI
    IOParallelXML)

  if (VTK_FOUND)
    message("VTK version: ${VTK_VERSION}")
  else()
    message ("VTK NOT FOUND")
  endif()
endif(PPKMHD_USE_VTK)

#####################################################################
# HDF5
#####################################################################
# prefer using parallel HDF5 when build with mpi
if (PPKMHD_USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif(PPKMHD_USE_MPI)

if (PPKMHD_USE_HDF5)
  find_package(HDF5)
  if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    set(my_hdf5_libs hdf5 hdf5_cpp)
    add_compile_options(-DUSE_HDF5)
    if (HDF5_IS_PARALLEL)
      add_compile_options(-DUSE_HDF5_PARALLEL)
    endif()
  endif(HDF5_FOUND)
endif(PPKMHD_USE_HDF5)

#####################################################################
# PNETCDF
#####################################################################
if (PPKMHD_USE_MPI)
  if (PPKMHD_USE_PNETCDF)
    find_package(PNETCDF)
    if (PNETCDF_FOUND)
      add_compile_options(-DUSE_PNETCDF)
      include_directories(${PNETCDF_INCLUDE_DIRS})
    endif(PNETCDF_FOUND)
  endif(PPKMHD_USE_PNETCDF)
endif(PPKMHD_USE_MPI)

#
# Kokkos : https://github.com/kokkos/kokkos
#
include(build_or_find_kokkos)

#####################################################################
# Eigen3
#####################################################################
# Eigen3 (for MOOD) -- not mature enought to be used with nvcc_wrapper,
# even on host code, I was not able to build code with nvcc_wrapper
#find_package(Eigen3)

#####################################################################
# Lapack with C interface --> OpenBLAS
#####################################################################
if (PPKMHD_USE_MOOD)
  # LAPACK / LAPACKE (for MOOD) - for DGEQRF and DTRSM (on host, not device)
  #set(BLA_VENDOR "OpenBLAS")
  #find_package(BLAS REQUIRED)
  find_package(OpenBLAS)
  find_package(LAPACKE)
  if ( NOT LAPACKE_FOUND )
    message(FATAL_ERROR "Package LAPACKE required, but not found!")
  endif( NOT LAPACKE_FOUND )
endif(PPKMHD_USE_MOOD)

#####################################################################
# PAPI for performance conuters
#####################################################################
find_package(PAPI)
if (NOT PAPI_FOUND)
  set(PAPI_INCLUDE_DIR "")
  set(PAPI_LIBRARY "")
endif(NOT PAPI_FOUND)

#
# common flags
#

if (Kokkos_ENABLE_CUDA)
  add_compile_options(-DCUDA)
endif()

if (PPKMHD_USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
endif()

if (PPKMHD_USE_MOOD)
  add_compile_options(-DUSE_MOOD)
endif()

if (PPKMHD_USE_SDM)
  add_compile_options(-DUSE_SDM)
endif()

if (PPKMHD_USE_MPI)
  add_compile_options(-DUSE_MPI)
endif()

if (PPKMHD_USE_FPE_DEBUG)
  add_compile_options(-DUSE_FPE_DEBUG)
endif()

##
## Using flags -Wextra, it's to strong for Kokkos, too many warnings
## But -Wall is really a minimum
##
#add_definitions(-Wall -Wextra -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wcast-qual -Wunused -Woverloaded-virtual  -pedantic )
#add_definitions( -Wall -Wextra )
add_definitions( -Wall )

#
# sources
#
# backtrace does not build anymore with cuda 10.1
#add_subdirectory(external/backward-cpp)

# pass Kokkos include directories to our target application
include_directories(${Kokkos_INCLUDE_DIRS_RET})

# CTest
enable_testing()

add_subdirectory(test)
add_subdirectory(src)

##################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("// ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("")
message("  CMake version          : ${CMAKE_VERSION}")
if (NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()
message("  CMake install prefix   : ${CMAKE_INSTALL_PREFIX}")
message("  CMake system processor : ${CMAKE_SYSTEM_PROCESSOR}")
message("  CMake system name (OS) : ${CMAKE_SYSTEM_NAME}")
message("")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
                         "${CMAKE_CXX_COMPILER_VERSION} "
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("                 ${CMAKE_CXX_COMPILER}")

message("  MPI enabled                  = ${PPKMHD_USE_MPI}")

message("  Kokkos version               = ${Kokkos_VERSION}")
message("  Kokkos_CXX_COMPILER          = ${Kokkos_CXX_COMPILER}")
message("  Kokkos_CXX_COMPILER_ID       = ${Kokkos_CXX_COMPILER_ID}")
message("  Kokkos_CXX_STANDARD          = ${Kokkos_CXX_STANDARD}")
message("  Kokkos_OPTIONS               = ${Kokkos_OPTIONS}")
message("  Kokkos_TPLS                  = ${Kokkos_TPLS}")
message("  Kokkos_DIRS                  = ${Kokkos_DIR}")

if(Kokkos_ENABLE_OPENMP)
  message("  Kokkos_ENABLE_OPENMP         = ${Kokkos_ENABLE_OPENMP}")
endif()

if(Kokkos_ENABLE_CUDA)
  message("  Kokkos_ENABLE_CUDA           = ${Kokkos_ENABLE_CUDA}")

  if( (${Kokkos_CUDA_LAMBDA_ENABLED}) OR (${Kokkos_ENABLE_CUDA_LAMBDA}))
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = OFF")
  endif()

  if( (${Kokkos_CUDA_CONSTEXPR_ENABLED}) OR (${Kokkos_ENABLE_CUDA_CONSTEXPR}))
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = OFF")
  endif()

  if( (${Kokkos_CUDA_UVM_ENABLED}) OR (${Kokkos_ENABLE_CUDA_UVM}))
    message("  Kokkos_ENABLE_CUDA_UVM       = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_UVM       = OFF")
  endif()

  message("  Kokkos CUDA   flags          = ${KOKKOS_CUDA_OPTIONS}")
  #message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
  #message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
  #message("  CUDA Compile flags : ${CUDA_NVCC_FLAGS}")
endif(Kokkos_ENABLE_CUDA)

if(Kokkos_ENABLE_HIP)
  message("  Kokkos_ENABLE_HIP            = ${Kokkos_ENABLE_HIP}")
endif(Kokkos_ENABLE_HIP)

if ( (${Kokkos_TPLS_HWLOC_ENABLED}) OR (${Kokkos_ENABLE_HWLOC}) )
  message("  Kokkos_ENABLE_HWLOC          = ON")
else()
  message("  Kokkos_ENABLE_HWLOC          = OFF")
endif()

message("  Kokkos architecture          = ${Kokkos_ARCH}")

if (HDF5_FOUND)
  message("  HDF5_VERSION                 = ${HDF5_VERSION}")
  message("  HDF5_DEFINITIONS             = ${HDF5_DEFINITIONS}")
  message("  HDF5_IS_PARALLEL             = ${HDF5_IS_PARALLEL}")
  message("  HDF5_INCLUDE_DIRS            = ${HDF5_INCLUDE_DIRS}")
  message("  HDF5_LIBRARIES               = ${HDF5_LIBRARIES}")
endif(HDF5_FOUND)

if (PNETCDF_FOUND)
  message("  PNETCDF_VERSION_STRING       = ${PNETCDF_VERSION_STRING}")
  message("  PNETCDF_INCLUDE_DIRS         = ${PNETCDF_INCLUDE_DIRS}")
  message("  PNETCDF_LIBRARIES            = ${PNETCDF_LIBRARIES}")
endif(PNETCDF_FOUND)

message("SDM      enabled : ${PPKMHD_USE_SDM}")
message("MOOD     enabled : ${PPKMHD_USE_MOOD}")
message("DOUBLE precision : ${PPKMHD_USE_DOUBLE}")

message("")
