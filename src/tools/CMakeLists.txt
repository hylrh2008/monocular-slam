find_package("OpenCV" REQUIRED)

add_executable(convert_dataset "convert_dataset.cpp")
target_link_libraries(convert_dataset ${OpenCV_LIBRARIES})
