init_test_directory()

create_test_custom_executable(file_stream_input_image_test "file_stream_input_image.cpp")
target_link_libraries(file_stream_input_image_test sdvo)
create_test(file_stream_input_image_test
  file_stream_input_image_test
  "${CMAKE_CURRENT_SOURCE_DIR}/file_stream_test"
#  "${CMAKE_SOURCE_DIR}/data/rgbd_dataset_freiburg1_xyz/rgb"
)
create_test_custom_executable(ssd_matcher_test "ssd_subpixel_matcher_over_line_test.cpp")
target_link_libraries(ssd_matcher_test sdvo dvo)
create_test(ssd_matcher_test
  ssd_matcher_test
  "${CMAKE_CURRENT_SOURCE_DIR}/ssd_matcher_test"
)
