cmake_minimum_required(VERSION 2.4.6)

add_definitions(-mssse3)

# Eigen3 library
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

# VTK library
find_package(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
   "${SSE_FLAGS} -O3 -march=native"
)

include_directories("include")

file(GLOB SOURCE
  "src/*.cpp"
  "src/core/*.cpp"
  "src/util/*.cpp"
  #"src/visualization/*.cpp"
)

file(GLOB HEADER
  "include/dvo/*.h*"
  "include/dvo/core/*.h*"
  "include/dvo/util/*.h*"
  #"include/dvo/visualization/*.h*"
)

add_library(${PROJECT_NAME} ${SOURCE} ${HEADER})
target_link_libraries(${PROJECT_NAME} tbb boost_thread)
