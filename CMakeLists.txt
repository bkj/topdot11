cmake_minimum_required(VERSION 3.6)
project(topdot)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")

find_package(OpenMP)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -mtune=native -march=native -Wall -Wunused-result -O3 -DNDEBUG -g -ffast-math")

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./pybind11/include
)

add_library(topdot SHARED topdot.cpp)

target_link_libraries(topdot ${PYTHON_LIBRARIES} OpenMP::OpenMP_CXX)
set_target_properties(topdot PROPERTIES PREFIX "")