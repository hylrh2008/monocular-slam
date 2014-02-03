project("sdvo")

find_package("OpenCV" REQUIRED)

file(GLOB HEADER "${CMAKE_SOURCE_DIR}/include/sdvo/*.h*")
file(GLOB SOURCE "*.cpp")

add_library(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

