add_subdirectory(kokkos)

add_subdirectory(mood)

add_subdirectory(lapack)

if(USE_MPI)
  add_subdirectory(mpiBasic)
endif(USE_MPI)

if (PAPI_FOUND)
  add_subdirectory(papi)
endif(PAPI_FOUND)
