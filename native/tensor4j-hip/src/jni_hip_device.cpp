#include <jni.h>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

static std::vector<std::string> deviceNames;
static std::vector<size_t> totalMems;

static void ensureDevices() {
    if (!deviceNames.empty()) return;
    int count;
    hipGetDeviceCount(&count);
    deviceNames.resize(count);
    totalMems.resize(count);
    for (int i = 0; i < count; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        deviceNames[i] = prop.name;
        totalMems[i] = prop.totalGlobalMem;
    }
}

extern "C" {

JNIEXPORT jstring JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nGetName(
    JNIEnv* env, jclass, jint ordinal) {
    ensureDevices();
    return env->NewStringUTF(deviceNames[ordinal].c_str());
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nGetTotalMemory(
    JNIEnv*, jclass, jint ordinal) {
    ensureDevices();
    return totalMems[ordinal];
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nGetFreeMemory(
    JNIEnv*, jclass, jint ordinal) {
    size_t free, total;
    hipMemGetInfo(&free, &total);
    return free;
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nMemAlloc(
    JNIEnv*, jclass, jlong bytes) {
    void* ptr;
    hipMalloc(&ptr, bytes);
    return reinterpret_cast<jlong>(ptr);
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nMemAllocManaged(
    JNIEnv*, jclass, jlong bytes) {
    void* ptr;
    hipMallocManaged(&ptr, bytes);
    return reinterpret_cast<jlong>(ptr);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nMemFree(
    JNIEnv*, jclass, jlong devPtr) {
    hipFree(reinterpret_cast<void*>(devPtr));
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nCompile(
    JNIEnv*, jclass, jstring source) {
    return 0;
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nModuleGetFunction(
    JNIEnv*, jclass, jlong modulePtr, jstring name) {
    return 0;
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nLaunchKernel(
    JNIEnv*, jclass, jlong funcPtr,
    jint gx, jint gy, jint gz, jint bx, jint by, jint bz,
    jlongArray bufPtrs, jintArray ints, jfloatArray floats) {
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nStreamCreate(JNIEnv*, jclass) {
    hipStream_t stream;
    hipStreamCreate(&stream);
    return reinterpret_cast<jlong>(stream);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nStreamSynchronize(
    JNIEnv*, jclass, jlong streamPtr) {
    hipStreamSynchronize(reinterpret_cast<hipStream_t>(streamPtr));
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nStreamDestroy(
    JNIEnv*, jclass, jlong streamPtr) {
    hipStreamDestroy(reinterpret_cast<hipStream_t>(streamPtr));
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_hip_HipDevice_nDeviceSynchronize(JNIEnv*, jclass) {
    hipDeviceSynchronize();
}

} // extern "C"
