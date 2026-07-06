#include <jni.h>
#include <cuda.h>
#include <nvrtc.h>
#include <string>
#include <vector>
#include <cstring>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nCompile(
    JNIEnv* env, jclass, jstring source) {
    const char* src = env->GetStringUTFChars(source, nullptr);

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src, "tensor4j_kernel.cu", 0, nullptr, nullptr);
    env->ReleaseStringUTFChars(source, src);

    const char* opts[] = {"--std=c++17", "--gpu-architecture=compute_50"};
    nvrtcResult compileRes = nvrtcCompileProgram(prog, 2, opts);

    if (compileRes != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      std::string("NVRTC compilation failed: ").append(log.data()).c_str());
        return 0;
    }

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    CUmodule module;
    cuModuleLoadData(&module, ptx.data());
    return reinterpret_cast<jlong>(module);
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nModuleGetFunction(
    JNIEnv*, jclass, jlong modulePtr, jstring name) {
    const char* kernelName = nullptr;
    kernelName = name ? nullptr : nullptr; // placeholder
    // In real impl: get string from JNI
    CUfunction func;
    // cuModuleGetFunction(&func, reinterpret_cast<CUmodule>(modulePtr), kernelName);
    return reinterpret_cast<jlong>(func);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nLaunchKernel(
    JNIEnv*, jclass, jlong funcPtr,
    jint gridX, jint gridY, jint gridZ,
    jint blockX, jint blockY, jint blockZ,
    jlongArray bufferPtrs, jintArray intArgs, jfloatArray floatArgs) {

    // Collect arguments into void* array
    // In real impl: set up kernel params and call cuLaunchKernel
    // cuLaunchKernel(reinterpret_cast<CUfunction>(funcPtr),
    //     gridX, gridY, gridZ, blockX, blockY, blockZ,
    //     0, nullptr, kernelParams, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nStreamCreate(
    JNIEnv*, jclass) {
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    return reinterpret_cast<jlong>(stream);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nStreamSynchronize(
    JNIEnv*, jclass, jlong streamPtr) {
    cuStreamSynchronize(reinterpret_cast<CUstream>(streamPtr));
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nStreamDestroy(
    JNIEnv*, jclass, jlong streamPtr) {
    cuStreamDestroy(reinterpret_cast<CUstream>(streamPtr));
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_cuda_CudaDevice_nDeviceSynchronize(
    JNIEnv*, jclass) {
    cudaDeviceSynchronize();
}

} // extern "C"
