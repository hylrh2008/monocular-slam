project("dvo_tester")
file(GLOB SOURCE "tracker.cpp")
add_executable(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} sdvo dvo lapack)

project("epipolar_matcher_tester")
file(GLOB SOURCE_EPIPOLAR "epipolar_matcher_test.cpp")
add_executable(${PROJECT_NAME} ${HEADER} ${SOURCE_EPIPOLAR})
target_link_libraries(${PROJECT_NAME} sdvo dvo lapack)
