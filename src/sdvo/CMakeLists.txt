project("sdvo")

find_package("OpenCV" REQUIRED)

file(GLOB HEADER "include/sdvo/*.h*")
file(GLOB SOURCE "src/*.cpp")

add_library(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

