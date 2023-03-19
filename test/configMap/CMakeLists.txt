#
# ConfigMap class test (src/util/config)
#

##############################################
add_executable(ConfigMapTest
  ConfigMapTest.cpp)

target_link_libraries(ConfigMapTest
  PRIVATE
  ppkMHD::config)

##############################################
if(PPKMHD_USE_MPI)

  configure_file(test_mpi.ini test_mpi.ini COPYONLY)

  add_executable(ConfigMapTest_mpi
    ConfigMapTest_mpi.cpp)

  target_link_libraries(ConfigMapTest_mpi
    PRIVATE
    ppkMHD::config
    ppkMHD::mpiUtils)

endif(PPKMHD_USE_MPI)
