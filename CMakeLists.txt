cmake_minimum_required(VERSION 1.9)

###########################
## SET PROJECT VARIABLES ##
###########################

# Where the resulting files will spawn.
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin"  )
set(LIBRARY_OUTPUT_PATH    "${CMAKE_SOURCE_DIR}/lib"  )
set(CMAKE_MODULE_PATH      "${CMAKE_SOURCE_DIR}/cmake")

# Build type:
# - None
# - Debug
# - Release
# - RelWithDebInfo
# - MinSizeRel
set(CMAKE_BUILD_TYPE Debug)

include("cmake/config.cmake")

# The main include directory of the project.
include_directories(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/3rd_party/dvo_core/include")

add_subdirectory("src")
