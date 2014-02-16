project("dvo_tester")

file(GLOB HEADER "*.h*")
file(GLOB SOURCE "*.cpp")

add_executable(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} sdvo dvo lapack)
