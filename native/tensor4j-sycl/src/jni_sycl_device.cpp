#include <jni.h>
#include <sycl/sycl.hpp>
#include <string>
#include <vector>

// Cache device properties per ordinal
static std::vector<sycl::device> cachedDevices;
static std::vector<std::string> deviceNames;
static std::vector<int> ccMajors;
static std::vector<int> ccMinors;
static std::vector<size_t> totalMems;

static void ensureDevices() {
    if (!cachedDevices.empty()) return;
    auto platforms = sycl::platform::get_platforms();
    // Enumerate all devices across all platforms
    for (auto& p : platforms) {
        auto devices = p.get_devices(sycl::info::device_type::all);
        for (auto& d : devices) {
            if (d.is_gpu()) {
                cachedDevices.push_back(d);
            }
        }
    }
    // Fallback: also collect all device types if no GPUs found
    if (cachedDevices.empty()) {
        for (auto& p : platforms) {
            auto devices = p.get_devices();
            for (auto& d : devices) {
                cachedDevices.push_back(d);
            }
        }
    }
    deviceNames.resize(cachedDevices.size());
    ccMajors.resize(cachedDevices.size());
    ccMinors.resize(cachedDevices.size());
    totalMems.resize(cachedDevices.size());
    for (size_t i = 0; i < cachedDevices.size(); i++) {
        auto& d = cachedDevices[i];
        deviceNames[i] = d.get_info<sycl::info::device::name>();
        // SYCL doesn't expose compute capability directly;
        // use version info as proxy
        auto ver = d.get_info<sycl::info::device::version>();
        // Parse "X.Y" string
        if (ver.size() >= 3 && ver[1] == '.') {
            ccMajors[i] = ver[0] - '0';
            ccMinors[i] = ver[2] - '0';
        } else {
            ccMajors[i] = 1;
            ccMinors[i] = 0;
        }
        totalMems[i] = d.get_info<sycl::info::device::global_mem_size>();
    }
}

extern "C" {

JNIEXPORT jstring JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nGetName(
    JNIEnv* env, jclass, jint ordinal) {
    ensureDevices();
    return env->NewStringUTF(deviceNames[ordinal].c_str());
}

JNIEXPORT jint JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nGetComputeCapabilityMajor(
    JNIEnv*, jclass, jint ordinal) {
    ensureDevices();
    return ccMajors[ordinal];
}

JNIEXPORT jint JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nGetComputeCapabilityMinor(
    JNIEnv*, jclass, jint ordinal) {
    ensureDevices();
    return ccMinors[ordinal];
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nGetTotalMemory(
    JNIEnv*, jclass, jint ordinal) {
    ensureDevices();
    return totalMems[ordinal];
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nGetFreeMemory(
    JNIEnv*, jclass, jint ordinal) {
    ensureDevices();
    auto& dev = cachedDevices[ordinal];
    // SYCL doesn't have a direct free-memory query; return total as estimate
    return static_cast<jlong>(dev.get_info<sycl::info::device::global_mem_size>());
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemAlloc(
    JNIEnv*, jclass, jlong bytes) {
    ensureDevices();
    // Allocate on first available GPU device
    sycl::queue q(cachedDevices[0]);
    void* ptr = sycl::malloc_device(bytes, q);
    return reinterpret_cast<jlong>(ptr);
}

JNIEXPORT jlong JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemAllocShared(
    JNIEnv*, jclass, jlong bytes) {
    ensureDevices();
    sycl::queue q(cachedDevices[0]);
    void* ptr = sycl::malloc_shared(bytes, q);
    return reinterpret_cast<jlong>(ptr);
}

JNIEXPORT void JNICALL Java_com_github_tensor4j_gpu_sycl_SyclDevice_nMemFree(
    JNIEnv*, jclass, jlong devPtr) {
    ensureDevices();
    sycl::queue q(cachedDevices[0]);
    sycl::free(reinterpret_cast<void*>(devPtr), q);
}

} // extern "C"
