#
# custom executable macro
#
macro(MPI_TEST_ADD_EXEC test_name)
  set(EXE_NAME ${test_name})
  add_executable(${EXE_NAME}
    ${EXE_NAME}.cpp
    $<TARGET_OBJECTS:time>
    $<TARGET_OBJECTS:mpiUtils>
    )
  
  target_include_directories(${EXE_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils
    )
endmacro(MPI_TEST_ADD_EXEC)


#
# add some executables
#
MPI_TEST_ADD_EXEC(testMpiHello)
MPI_TEST_ADD_EXEC(testMpiTopology_2D_C_API)
MPI_TEST_ADD_EXEC(testMpiTopology_2D_Cpp_API)
MPI_TEST_ADD_EXEC(testMpiTopology_3D_Cpp_API)
MPI_TEST_ADD_EXEC(testMpiHelloGlobalSession)
MPI_TEST_ADD_EXEC(testMpiTopology_2D)
MPI_TEST_ADD_EXEC(testMpiTopology_3D_C_API)
#testVtkXMLPImageDataWriter
