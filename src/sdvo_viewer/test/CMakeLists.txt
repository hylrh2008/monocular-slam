init_test_directory()

create_test_executable(config_handler_test "config_handler_test.cpp" "../config_handler.cpp")
target_link_libraries(config_handler_test ${Boost_LIBRARIES})
create_test(config_handler_test config_handler_test)
