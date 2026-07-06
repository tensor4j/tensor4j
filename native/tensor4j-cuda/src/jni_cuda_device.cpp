#include <jni.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string>
#include <vector>

static std::vector<std::string> deviceNames;
static std::vector<int> ccMajors;
static std::vector<int> ccMinors;
static std::vector<size_t> totalMems;

static void ensureDevices() {
    if (!deviceNames.empty()) return;
    int count;
    cudaGetDeviceCount(&count);
    deviceNames.resize(count);
    ccMajors.resize(count);
    ccMinors.resize(count);
    totalMems.resize(count);
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        deviceNames[i] = prop.name;
        ccMajors[i] = prop.major;
        ccMinors[i] = prop.minor;
        totalMems[i] = prop.totalGlobalMem;
    }
}

extern "C" {

JNIEXPORT jstring JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nGetName(
    JNIEnv* env, jclass, jint ordinal) {
    ensureDevices();
    return env->NewStringUTF(deviceNames[ordinal].c_str());
}

JNIEXPORT jint JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nGetComputeCapabilityMajor(
    JNIEnv*, jclass, jint ordinal) {
    ensureDevices();
    return ccMajors[ordinal];
}

JNIEXPORT jint JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nGetComputeCapabilityMinor(
    JNIEnv*, jclass, jint ordinal) {
    ensureDevices();
    return ccMinors[ordinal];
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nGetTotalMemory(
    JNIEnv*, jclass, jint ordinal) {
    ensureDevices();
    return totalMems[ordinal];
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nGetFreeMemory(
    JNIEnv*, jclass, jint ordinal) {
    cudaSetDevice(ordinal);
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nMemAlloc(
    JNIEnv*, jclass, jlong bytes) {
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return reinterpret_cast<jlong>(ptr);
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nMemAllocManaged(
    JNIEnv*, jclass, jlong bytes) {
    void* ptr;
    cudaMallocManaged(&ptr, bytes);
    return reinterpret_cast<jlong>(ptr);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nMemFree(
    JNIEnv*, jclass, jlong devPtr) {
    cudaFree(reinterpret_cast<void*>(devPtr));
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nMemcpyHtoD(
    JNIEnv* env, jclass, jlong dstDevPtr, jfloatArray src, jlong srcOff, jlong count) {
    jfloat* srcPtr = env->GetFloatArrayElements(src, nullptr);
    cudaMemcpy(reinterpret_cast<void*>(dstDevPtr),
               srcPtr + srcOff, count * sizeof(float), cudaMemcpyHostToDevice);
    env->ReleaseFloatArrayElements(src, srcPtr, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nMemcpyDtoH(
    JNIEnv* env, jclass, jfloatArray dst, jlong dstOff, jlong srcDevPtr, jlong srcOff, jlong count) {
    jfloat* dstPtr = env->GetFloatArrayElements(dst, nullptr);
    cudaMemcpy(dstPtr + dstOff,
               reinterpret_cast<void*>(srcDevPtr) + srcOff * sizeof(float),
               count * sizeof(float), cudaMemcpyDeviceToHost);
    env->ReleaseFloatArrayElements(dst, dstPtr, 0);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nMemcpyDtoD(
    JNIEnv*, jclass, jlong dstDevPtr, jlong dstOff, jlong srcDevPtr, jlong srcOff, jlong bytes) {
    cudaMemcpy(reinterpret_cast<void*>(dstDevPtr + dstOff),
               reinterpret_cast<void*>(srcDevPtr + srcOff),
               bytes, cudaMemcpyDeviceToDevice);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nMemset(
    JNIEnv*, jclass, jlong devPtr, jbyte val, jlong bytes) {
    cudaMemset(reinterpret_cast<void*>(devPtr), val, bytes);
}

} // extern "C"
