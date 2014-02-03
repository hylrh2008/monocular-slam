cmake_minimum_required(VERSION 1.9)

###########################
## SET PROJECT VARIABLES ##
###########################

# Where the resulting files will spawn.
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH    "${CMAKE_SOURCE_DIR}/lib")

# Build type:
# - None
# - Debug
# - Release
# - RelWithDebInfo
# - MinSizeRel
set(CMAKE_BUILD_TYPE Debug)

# The main include directory of the project.
include_directories("${CMAKE_SOURCE_DIR}/include")

add_subdirectory("src")
