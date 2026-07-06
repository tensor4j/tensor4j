#include <jni.h>
#include <sycl/sycl.hpp>
#include <vector>

// Shared queue for buffer operations (lazily created)
static sycl::queue& getQueue() {
    static sycl::queue q;
    return q;
}

extern "C" {

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemcpyHtoD(
    JNIEnv* env, jclass, jlong dstDevPtr, jfloatArray src, jlong srcOff, jlong dstOff, jlong count) {
    jfloat* srcPtr = env->GetFloatArrayElements(src, nullptr);
    auto& q = getQueue();
    q.memcpy(reinterpret_cast<void*>(dstDevPtr + dstOff * sizeof(float)),
             srcPtr + srcOff,
             count * sizeof(float));
    q.wait();
    env->ReleaseFloatArrayElements(src, srcPtr, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemcpyHtoDAsync(
    JNIEnv* env, jclass, jlong dstDevPtr, jfloatArray src, jlong srcOff, jlong dstOff, jlong count, jlong queuePtr) {
    jfloat* srcPtr = env->GetFloatArrayElements(src, nullptr);
    auto q = *reinterpret_cast<sycl::queue*>(queuePtr);
    q.memcpy(reinterpret_cast<void*>(dstDevPtr + dstOff * sizeof(float)),
             srcPtr + srcOff,
             count * sizeof(float));
    env->ReleaseFloatArrayElements(src, srcPtr, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemcpyDtoH(
    JNIEnv* env, jclass, jfloatArray dst, jlong dstOff, jlong srcDevPtr, jlong srcOff, jlong count) {
    jfloat* dstPtr = env->GetFloatArrayElements(dst, nullptr);
    auto& q = getQueue();
    q.memcpy(dstPtr + dstOff,
             reinterpret_cast<void*>(srcDevPtr + srcOff * sizeof(float)),
             count * sizeof(float));
    q.wait();
    env->ReleaseFloatArrayElements(dst, dstPtr, 0);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemcpyDtoHAsync(
    JNIEnv* env, jclass, jfloatArray dst, jlong dstOff, jlong srcDevPtr, jlong srcOff, jlong count, jlong queuePtr) {
    jfloat* dstPtr = env->GetFloatArrayElements(dst, nullptr);
    auto q = *reinterpret_cast<sycl::queue*>(queuePtr);
    q.memcpy(dstPtr + dstOff,
             reinterpret_cast<void*>(srcDevPtr + srcOff * sizeof(float)),
             count * sizeof(float));
    env->ReleaseFloatArrayElements(dst, dstPtr, 0);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemcpyDtoD(
    JNIEnv*, jclass, jlong dstDevPtr, jlong dstOff, jlong srcDevPtr, jlong srcOff, jlong bytes, jlong queuePtr) {
    auto q = *reinterpret_cast<sycl::queue*>(queuePtr);
    q.memcpy(reinterpret_cast<void*>(dstDevPtr + dstOff),
             reinterpret_cast<void*>(srcDevPtr + srcOff),
             bytes);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemset(
    JNIEnv*, jclass, jlong devPtr, jbyte val, jlong bytes) {
    auto& q = getQueue();
    q.memset(reinterpret_cast<void*>(devPtr), val, bytes);
    q.wait();
}

} // extern "C"
