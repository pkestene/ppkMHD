add_subdirectory(config)
add_subdirectory(monitoring)
add_subdirectory(io)
if (PPKMHD_USE_MPI)
  add_subdirectory(mpiUtils)
endif()
