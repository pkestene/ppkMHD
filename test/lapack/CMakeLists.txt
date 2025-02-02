#
# Lapacked related tests
#

# ##################################################################################################
add_executable(test_lapacke test_lapacke.cpp)
target_include_directories(test_lapacke PUBLIC ${LAPACKE_INCLUDE_DIR})
target_link_libraries(test_lapacke ${LAPACKE_LIBRARIES})

# ##################################################################################################
add_executable(test_lapacke_qr test_lapacke_qr.cpp)
target_include_directories(test_lapacke_qr PUBLIC ${LAPACKE_INCLUDE_DIR} ${OpenBLAS_INCLUDE_DIR})
# target_link_libraries(test_lapacke_qr ${LAPACKE_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(test_lapacke_qr ${LAPACKE_LIBRARIES} ${OpenBLAS_LIB})
