add_subdirectory(kokkos)

add_subdirectory(mood)

if(USE_MPI)
  add_subdirectory(mpiBasic)
endif(USE_MPI)
