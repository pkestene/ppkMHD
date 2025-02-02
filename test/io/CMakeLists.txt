configure_file(test_io_2d.ini test_io_2d.ini COPYONLY)
configure_file(test_io_3d.ini test_io_3d.ini COPYONLY)

################################
# VTK test
################################
add_executable(test_io_vtk "")

target_sources(test_io_vtk
  PUBLIC
  test_io_vtk.cpp)

# target_include_directories(test_io_vtk
#   PUBLIC
#   ${CMAKE_SOURCE_DIR}/src
#   ${CMAKE_SOURCE_DIR}/src/utils/io
#   ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils
#   )
target_link_libraries(test_io_vtk
  PUBLIC
  kokkos
  ppkMHD::shared
  ppkMHD::config
  ppkMHD::monitoring
  ppkMHD::io)

if(PPKMHD_USE_MPI)
  target_link_libraries(test_io_vtk
    PUBLIC
    ppkMHD::mpiUtils)
endif(PPKMHD_USE_MPI)

###############################
# HDF5 test
###############################
if(PPKMHD_USE_HDF5)
  add_executable(test_io_hdf5
    test_io_hdf5.cpp
    )

  target_link_libraries(test_io_hdf5
    PUBLIC
    kokkos
    ppkMHD::shared
    ppkMHD::config
    ppkMHD::monitoring
    ppkMHD::io
    ${HDF5_LIBRARIES})

  if(PPKMHD_USE_MPI)
    target_link_libraries(test_io_hdf5
      PUBLIC
      ppkMHD::mpiUtils)
  endif(PPKMHD_USE_MPI)

endif(PPKMHD_USE_HDF5)

if(PPKMHD_USE_MPI)
  if(PPKMHD_USE_PNETCDF)

    add_executable(test_io_pnetcdf
      test_io_pnetcdf.cpp
      )

    target_link_libraries(test_io_pnetcdf
      PUBLIC
      kokkos
      ppkMHD::shared
      ppkMHD::config
      ppkMHD::monitoring
      ppkMHD::io
      ppkMHD::mpiUtils
      ${PNETCDF_LIBRARIES})

  endif(PPKMHD_USE_PNETCDF)
endif(PPKMHD_USE_MPI)
