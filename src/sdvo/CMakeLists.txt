project("sdvo")

find_package("OpenCV" REQUIRED)
find_package("Boost" COMPONENTS "filesystem" "system" "regex" REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB HEADER "${CMAKE_SOURCE_DIR}/include/sdvo/*.h*")
file(GLOB SOURCE "*.cpp")
#set_source_files_properties(ssd_subpixel_matcher_over_line.cpp PROPERTIES COMPILE_FLAGS "-O3 -msse -msse2 -msse3")

add_library(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_test_directory("test")
