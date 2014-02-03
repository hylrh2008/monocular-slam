project("sdvo_viewer")

find_package("Boost" COMPONENTS "program_options" REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ".")

file(GLOB HEADER "*.h*")
file(GLOB SOURCE "*.cpp")

add_executable(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} sdvo)

add_test_directory("test")
