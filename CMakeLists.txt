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

# OpenMP
find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# The main include directory of the project.
include_directories(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/3rd_party/dvo_core/include"
  "${CMAKE_SOURCE_DIR}/3rd_party/sophus")

include("cmake/config.cmake")

add_subdirectory("src")
