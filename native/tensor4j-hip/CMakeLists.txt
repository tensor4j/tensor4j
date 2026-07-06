cmake_minimum_required(VERSION 3.18)
project(tensor4j-hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)
find_package(hip REQUIRED)

add_library(tensor4j-hip SHARED
    src/jni_hip_device.cpp
)

target_include_directories(tensor4j-hip PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${hip_INCLUDE_DIRS}
)

target_link_libraries(tensor4j-hip PRIVATE
    ${hip_LIBRARIES}
)
