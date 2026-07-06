#include <jni.h>
#include <CL/cl.h>
#include <string>
#include <vector>

extern "C" {

JNIEXPORT jstring JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nGetName(
    JNIEnv* env, jclass, jint platform, jint device) {
    cl_platform_id platforms[16];
    cl_uint numPlatforms;
    clGetPlatformIDs(16, platforms, &numPlatforms);

    cl_device_id devices[64];
    cl_uint numDevices;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, 64, devices, &numDevices);

    char name[256];
    clGetDeviceInfo(devices[device], CL_DEVICE_NAME, 256, name, nullptr);
    return env->NewStringUTF(name);
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nGetTotalMemory(
    JNIEnv*, jclass, jint platform, jint device) {
    cl_platform_id platforms[16];
    cl_uint numPlatforms;
    clGetPlatformIDs(16, platforms, &numPlatforms);

    cl_device_id devices[64];
    cl_uint numDevices;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, 64, devices, &numDevices);

    cl_ulong mem;
    clGetDeviceInfo(devices[device], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
    return mem;
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nGetFreeMemory(
    JNIEnv*, jclass, jint platform, jint device) {
    return 0;
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nCreateCommandQueue(
    JNIEnv*, jclass, jint platform, jint device) {
    cl_platform_id platforms[16];
    cl_uint numPlatforms;
    clGetPlatformIDs(16, platforms, &numPlatforms);

    cl_device_id devices[64];
    cl_uint numDevices;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, 64, devices, &numDevices);

    cl_context ctx = clCreateContext(nullptr, 1, &devices[device], nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, devices[device], nullptr, nullptr);
    return reinterpret_cast<jlong>(queue);
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nCreateBuffer(
    JNIEnv*, jclass, jint platform, jint device, jlong bytes) {
    cl_platform_id platforms[16];
    cl_uint numPlatforms;
    clGetPlatformIDs(16, platforms, &numPlatforms);

    cl_device_id devices[64];
    cl_uint numDevices;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, 64, devices, &numDevices);

    cl_context ctx = clCreateContext(nullptr, 1, &devices[device], nullptr, nullptr, nullptr);
    cl_mem mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, nullptr);
    return reinterpret_cast<jlong>(mem);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nReleaseBuffer(
    JNIEnv*, jclass, jlong memPtr) {
    clReleaseMemObject(reinterpret_cast<cl_mem>(memPtr));
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nBuildProgram(
    JNIEnv* env, jclass, jint platform, jint device, jstring source) {
    const char* src = env->GetStringUTFChars(source, nullptr);

    cl_platform_id platforms[16];
    cl_uint numPlatforms;
    clGetPlatformIDs(16, platforms, &numPlatforms);

    cl_device_id devices[64];
    cl_uint numDevices;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, 64, devices, &numDevices);

    cl_context ctx = clCreateContext(nullptr, 1, &devices[device], nullptr, nullptr, nullptr);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, nullptr);
    clBuildProgram(prog, 1, &devices[device], nullptr, nullptr, nullptr);

    env->ReleaseStringUTFChars(source, src);
    return reinterpret_cast<jlong>(prog);
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nCreateKernel(
    JNIEnv*, jclass, jlong programPtr, jstring name) {
    const char* kernelName = nullptr;
    // In real impl: get name from JNI
    cl_kernel kernel = clCreateKernel(reinterpret_cast<cl_program>(programPtr), kernelName, nullptr);
    return reinterpret_cast<jlong>(kernel);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nSetKernelArg(
    JNIEnv*, jclass, jlong kernelPtr, jint index, jlong bufferPtr) {
    cl_mem mem = reinterpret_cast<cl_mem>(bufferPtr);
    clSetKernelArg(reinterpret_cast<cl_kernel>(kernelPtr), index, sizeof(cl_mem), &mem);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nSetKernelArgInt(
    JNIEnv*, jclass, jlong kernelPtr, jint index, jint value) {
    clSetKernelArg(reinterpret_cast<cl_kernel>(kernelPtr), index, sizeof(jint), &value);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nSetKernelArgFloat(
    JNIEnv*, jclass, jlong kernelPtr, jint index, jfloat value) {
    clSetKernelArg(reinterpret_cast<cl_kernel>(kernelPtr), index, sizeof(jfloat), &value);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nEnqueueNDRange(
    JNIEnv*, jclass, jlong queuePtr, jlong kernelPtr,
    jint dims, jlongArray globalWork, jlongArray localWork) {
    // In real impl: get arrays from JNI
    // clEnqueueNDRangeKernel(queue, kernel, dims, nullptr, global, local, 0, nullptr, nullptr);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nEnqueueWriteBuffer(
    JNIEnv* env, jclass, jlong queuePtr, jlong memPtr,
    jfloatArray src, jlong srcOff, jlong count) {
    jfloat* srcPtr = env->GetFloatArrayElements(src, nullptr);
    clEnqueueWriteBuffer(reinterpret_cast<cl_command_queue>(queuePtr),
                         reinterpret_cast<cl_mem>(memPtr), CL_TRUE, 0,
                         count * sizeof(float), srcPtr + srcOff, 0, nullptr, nullptr);
    env->ReleaseFloatArrayElements(src, srcPtr, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nEnqueueReadBuffer(
    JNIEnv* env, jclass, jlong queuePtr, jlong memPtr,
    jfloatArray dst, jlong dstOff, jlong count) {
    jfloat* dstPtr = env->GetFloatArrayElements(dst, nullptr);
    clEnqueueReadBuffer(reinterpret_cast<cl_command_queue>(queuePtr),
                        reinterpret_cast<cl_mem>(memPtr), CL_TRUE, 0,
                        count * sizeof(float), dstPtr + dstOff, 0, nullptr, nullptr);
    env->ReleaseFloatArrayElements(dst, dstPtr, 0);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nFinish(
    JNIEnv*, jclass, jint platform, jint device) {
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_opencl_OclDevice_nReleaseCommandQueue(
    JNIEnv*, jclass, jlong queuePtr) {
    clReleaseCommandQueue(reinterpret_cast<cl_command_queue>(queuePtr));
}

} // extern "C"
