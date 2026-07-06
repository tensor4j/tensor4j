cmake_minimum_required(VERSION 3.18)
project(tensor4j-opencl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)
find_package(OpenCL REQUIRED)

add_library(tensor4j-opencl SHARED
    src/jni_ocl_device.cpp
)

target_include_directories(tensor4j-opencl PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(tensor4j-opencl PRIVATE
    ${OpenCL_LIBRARIES}
)
