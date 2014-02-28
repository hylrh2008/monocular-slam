project("dvo_tester")
file(GLOB SOURCE "tracker.cpp")
add_executable(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} sdvo dvo lapack)

project("epipolar_matcher_tester")
file(GLOB SOURCE_EPIPOLAR "epipolar_matcher_test.cpp")
find_package(PCL REQUIRED COMPONENTS common io visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} ${HEADER} ${SOURCE_EPIPOLAR})
target_link_libraries(${PROJECT_NAME} sdvo dvo lapack ${PCL_COMMON_LIBRARIES}
                      ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})
