#
# Kokkos test example : query device information
#

add_executable(query_device_kokkos
  query_device_kokkos.cpp)
set_target_properties(query_device_kokkos PROPERTIES
  COMPILE_FLAGS "${KOKKOS_CPPFLAGS} ${KOKKOS_CXXFLAGS}")
set_target_properties(query_device_kokkos PROPERTIES
  LINK_FLAGS "${KOKKOS_LDFLAGS}")
target_link_libraries(query_device_kokkos kokkos hwloc)
