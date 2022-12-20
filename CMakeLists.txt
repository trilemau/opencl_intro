cmake_minimum_required(VERSION 3.1)
project(OpenCL_intro)

# Configure compiler warnings
if (CMAKE_CXX_COMPILER_ID MATCHES Clang OR ${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
  # using regular Clang or AppleClang or GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -pedantic -Wold-style-cast")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS "/std:c++17 /permissive- /W4 /EHsc")
  add_definitions("-D_CRT_SECURE_NO_WARNINGS")
endif()

set(SOURCES )
set(TEST_SOURCES test-main.cpp)

add_executable(${PROJECT_NAME} main.cpp ${SOURCES})
add_executable(tests ${TEST_SOURCES} ${SOURCES})

# Configure OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (${PROJECT_NAME} ${OpenCL_LIBRARY})
