
#################################
## BUILD FLAGS
#################################


# Use C++11
set(CMAKE_CXX_FLAGS          "${CMAKE_CXX_FLAGS} -std=gnu++0x"                        )

# Set Coverage flags.
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS_COVERAGE   "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage"  )
if ("${CMAKE_BUILD_TYPE}" STREQUAL COVERAGE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

#################################
## TEST FACILITIES
#################################


macro(ADD_TEST_DIRECTORY _directory)
  if (BUILD_TESTS)
    add_subdirectory(${_directory})
  endif()
endmacro()

macro(INIT_TEST_DIRECTORY)
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/test)
  set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib/test)
  include(Dart)
endmacro()

macro(CREATE_TEST_EXECUTABLE _name)
  add_executable(${_name} ${ARGN})
  target_link_libraries(${_name} gtest_main pthread)
endmacro()

macro(CREATE_TEST_CUSTOM_EXECUTABLE _name)
  add_executable(${_name} ${ARGN})
  target_link_libraries(${_name} gtest pthread)
endmacro()

macro(CREATE_TEST _name _executable)
  add_test(NAME ${_name} COMMAND ${EXECUTABLE_OUTPUT_PATH}/${_executable} ${ARGN})
endmacro()

macro(CREATE_TEST_CUSTOM _name _executable)
  add_test(NAME ${_name} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/${_executable} ${ARGN})
endmacro()

if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(gtest)
  include_directories(gtest/include)
endif()


#################################
## ADD THIRD PARTY
#################################


add_subdirectory(3rd_party)

file(GLOB THIRD_PARTY_INC_DIRS "3rd_party/*/include")
include_directories(${THIRD_PARTY_INC_DIRS})


