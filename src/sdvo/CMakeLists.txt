project("sdvo")


find_package("OpenCV" REQUIRED)
find_package("Boost" COMPONENTS "filesystem" "system" "regex" REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB HEADER "${CMAKE_SOURCE_DIR}/include/sdvo/*.h*")
file(GLOB SOURCE "*.cpp")

add_library(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_test_directory("test")
