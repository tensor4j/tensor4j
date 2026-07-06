cmake_minimum_required(VERSION 3.18)
project(tensor4j-cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(tensor4j-cuda SHARED
    src/jni_cuda_device.cpp
    src/jni_cuda_buffer.cpp
    src/jni_cuda_kernel.cpp
    src/jni_cuda_stream.cpp
)

target_include_directories(tensor4j-cuda PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tensor4j-cuda PRIVATE
    ${CUDAToolkit_LIBRARIES}
)
