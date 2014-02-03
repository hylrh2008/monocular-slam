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

add_library(${PROJECT_NAME}
    src/core/interpolation.cpp
    src/core/intrinsic_matrix.cpp
    src/core/least_squares.cpp
    src/core/math_sse.cpp
    src/core/rgbd_image.cpp
    src/core/rgbd_image_sse.cpp
    src/core/surface_pyramid.cpp
    src/core/weight_calculation.cpp

    src/util/histogram.cpp

    src/visualization/async_point_cloud_builder.cpp
    src/visualization/point_cloud_aggregator.cpp
    src/visualization/visualizer.cpp
    src/visualization/camera_trajectory_visualizer.cpp
    src/visualization/pcl_camera_trajetory_visualizer.cpp

    src/dense_tracking.cpp
    src/dense_tracking_config.cpp
)

target_link_libraries(${PROJECT_NAME}
    tbb
    boost_thread
)
