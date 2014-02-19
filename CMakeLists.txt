cmake_minimum_required(VERSION 2.8)

###########################
## SET PROJECT VARIABLES ##
###########################

# Where the resulting files will spawn.
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin-${CMAKE_BUILD_TYPE}"  )
set(LIBRARY_OUTPUT_PATH    "${CMAKE_SOURCE_DIR}/lib-${CMAKE_BUILD_TYPE}"  )
set(CMAKE_MODULE_PATH      "${CMAKE_SOURCE_DIR}/cmake")

option(BUILD_TESTS "Build the tests." ON)

# Build type:
# - None
# - Debug
# - Release
# - RelWithDebInfo
# - MinSizeRel
#set(CMAKE_BUILD_TYPE RELEASE)
add_definitions(-mssse3 -march=native)

# Eigen3 library
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

# The main include directory of the project.
include_directories(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/3rd_party/dvo_core/include"
  "${CMAKE_SOURCE_DIR}/3rd_party/sophus")

include("cmake/config.cmake")

add_subdirectory("src")
