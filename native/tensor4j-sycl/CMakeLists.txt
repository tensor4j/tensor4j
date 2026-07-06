cmake_minimum_required(VERSION 3.18)
project(tensor4j-sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)

# SYCL implementation: oneAPI DPC++ uses find_package(IntelDPCPP) or IntelSYCL
# AdaptiveCpp (hipSYCL) uses find_package(AdaptiveCpp)
# For generic builds, set SYCL_INCLUDE_DIR and SYCL_LIBRARY manually
if(DEFINED ENV{ONEAPI_ROOT})
    # Intel oneAPI DPC++
    set(SYCL_INCLUDE_DIR "$ENV{ONEAPI_ROOT}/compiler/latest/linux/include/sycl")
    set(SYCL_LIBRARY "$ENV{ONEAPI_ROOT}/compiler/latest/linux/lib/libsycl.so")
elseif(DEFINED ENV{SYCL_INCLUDE_DIR})
    # User-provided paths
else()
    message(FATAL_ERROR "SYCL not found. Set ONEAPI_ROOT or SYCL_INCLUDE_DIR and SYCL_LIBRARY")
endif()

add_library(tensor4j-sycl SHARED
    src/jni_sycl_device.cpp
    src/jni_sycl_buffer.cpp
    src/jni_sycl_kernel.cpp
    src/jni_sycl_stream.cpp
)

target_include_directories(tensor4j-sycl PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${SYCL_INCLUDE_DIR}
)

target_link_libraries(tensor4j-sycl PRIVATE
    ${SYCL_LIBRARY}
)
