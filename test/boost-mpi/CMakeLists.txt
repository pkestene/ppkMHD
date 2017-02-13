#
# MPI-boost example
#

add_executable( hello_world
  hello_world.cpp )
set_target_properties( hello_world PROPERTIES
  COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
include_directories( ${Boost_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH} )
#set_target_properties( hello_world PROPERTIES
#  LINK_FLAGS "${MPI_CXX_LINK_FLAGS} -L${Boost_LIBRARY_DIR_RELEASE}")
target_link_libraries( hello_world ${Boost_MPI_LIBRARY_RELEASE} ${Boost_SERIALIZATION_LIBRARY_RELEASE} ${MPI_CXX_LIBRARIES})
