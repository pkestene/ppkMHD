add_subdirectory(kokkos)
add_subdirectory(shared)

if(PPKMHD_USE_MOOD)
  add_subdirectory(mood)

  add_subdirectory(lapack)
endif(PPKMHD_USE_MOOD)

# backtrace does not build anymore with cuda 10.1
#add_subdirectory(backward-cpp)
add_subdirectory(io)
add_subdirectory(configMap)

if(PPKMHD_USE_MPI)
  add_subdirectory(mpiBasic)
endif(PPKMHD_USE_MPI)

if(PAPI_FOUND)
  add_subdirectory(papi)
endif(PAPI_FOUND)

if(PPKMHD_USE_SDM)
  add_subdirectory(sdm)
endif(PPKMHD_USE_SDM)
