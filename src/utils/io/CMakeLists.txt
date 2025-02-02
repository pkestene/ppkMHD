add_library(io "")

target_sources(
  io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/IO_common.cpp ${CMAKE_CURRENT_SOURCE_DIR}/IO_ReadWrite.cpp
             ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.cpp)

if(PPKMHD_USE_SDM)
  target_sources(
    io
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/IO_ReadWrite_SDM.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK_SDM_shared.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK_SDM.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK_SDM_Flux.cpp)
endif(PPKMHD_USE_SDM)

if(PPKMHD_USE_HDF5)
  target_sources(io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/IO_HDF5.cpp)
endif(PPKMHD_USE_HDF5)

if(PPKMHD_USE_MPI)
  if(PPKMHD_USE_PNETCDF)
    target_sources(io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/IO_PNETCDF.h)
  endif(PPKMHD_USE_PNETCDF)
endif(PPKMHD_USE_MPI)

target_include_directories(io PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(io PUBLIC kokkos shared)

if(PPKMHD_USE_HDF5)
  target_link_libraries(io PUBLIC ${HDF5_LIBRARIES})
endif(PPKMHD_USE_HDF5)

if(PPKMHD_USE_PNETCDF)
  target_link_libraries(io PUBLIC ${PNETCDF_LIBRARIES})
endif(PPKMHD_USE_PNETCDF)

if(PPKMHD_USE_MPI)
  target_link_libraries(io PUBLIC MPI::MPI_CXX)
endif(PPKMHD_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(ppkMHD::io ALIAS io)
