add_custom_target(copy_files ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/sphinx" "${CMAKE_CURRENT_BINARY_DIR}/sphinx" )

