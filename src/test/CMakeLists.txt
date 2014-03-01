project("dvo_tester")
file(GLOB SOURCE "tracker.cpp")
add_executable(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} sdvo dvo lapack)

project("epipolar_matcher_tester")
file(GLOB SOURCE_EPIPOLAR "epipolar_matcher_test.cpp")

find_package(PCL COMPONENTS common io visualization)
if(PCL_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  add_definitions(-D_ENABLE_PCL)
endif(PCL_FOUND)

add_executable(${PROJECT_NAME} ${HEADER} ${SOURCE_EPIPOLAR})
target_link_libraries(${PROJECT_NAME} sdvo dvo lapack ${PCL_COMMON_LIBRARIES}
                      ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})
