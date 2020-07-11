cmake_minimum_required(VERSION 3.16)
project(Symdetect)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
add_executable(symdetect src/main.cpp src/image_stacker.cpp src/symdetect.cpp src/geom.cpp)
target_include_directories(symdetect PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(symdetect ${OpenCV_LIBS})
