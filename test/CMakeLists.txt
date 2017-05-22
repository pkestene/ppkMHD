add_subdirectory(kokkos)

if(USE_MOOD)
  add_subdirectory(mood)

  add_subdirectory(lapack)
endif()

add_subdirectory(io)

if(USE_MPI)
  add_subdirectory(mpiBasic)
endif(USE_MPI)

if(PAPI_FOUND)
  add_subdirectory(papi)
endif(PAPI_FOUND)
