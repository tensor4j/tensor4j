#include <jni.h>
#include <sycl/sycl.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nStreamCreate(
    JNIEnv*, jclass) {
    // SYCL uses sycl::queue as the stream abstraction
    sycl::queue* q = new sycl::queue();
    return reinterpret_cast<jlong>(q);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nStreamSynchronize(
    JNIEnv*, jclass, jlong queuePtr) {
    auto* q = reinterpret_cast<sycl::queue*>(queuePtr);
    q->wait();
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nStreamDestroy(
    JNIEnv*, jclass, jlong queuePtr) {
    auto* q = reinterpret_cast<sycl::queue*>(queuePtr);
    q->wait();
    delete q;
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nDeviceSynchronize(
    JNIEnv*, jclass) {
    sycl::queue().wait();
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nEventCreate(
    JNIEnv*, jclass) {
    // SYCL events are lightweight; store 1 as a dummy
    return 1;
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nEventDestroy(
    JNIEnv*, jclass, jlong) {
    // No-op
}

} // extern "C"
