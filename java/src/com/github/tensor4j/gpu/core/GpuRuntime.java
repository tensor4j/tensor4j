/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.core;

import com.github.tensor4j.gpu.ref.CpuDevice;

/**
 * GPU runtime detection and device selection.
 * Auto-selects best available device: CUDA > HIP > SYCL > OpenCL > CPU fallback.
 */
public final class GpuRuntime {

    private static GpuDevice defaultDevice;
    private static boolean detectionAttempted;

    private GpuRuntime() {
    }

    public static synchronized GpuDevice defaultDevice() {
        if (!detectionAttempted) {
            defaultDevice = detectBestDevice();
            detectionAttempted = true;
        }
        return defaultDevice;
    }

    public static synchronized void setDefaultDevice(GpuDevice device) {
        if (defaultDevice != null && defaultDevice != device) {
            try {
                defaultDevice.close();
            } catch (Exception ignored) {
            }
        }
        defaultDevice = device;
        detectionAttempted = true;
    }

    public static GpuTensorMath math() {
        return new GpuTensorMath(defaultDevice());
    }

    private static GpuDevice detectBestDevice() {
        try {
            Class<?> cudaCls = Class.forName("com.github.tensor4j.gpu.cuda.CudaDevice");
            GpuDevice cuda = (GpuDevice) cudaCls.getDeclaredConstructor().newInstance();
            if (cuda.computeCapabilityMajor() >= 5) {
                return cuda;
            }
            try {
                cuda.close();
            } catch (Exception ignored) {
            }
        } catch (Exception ignored) {
        }

        try {
            Class<?> hipCls = Class.forName("com.github.tensor4j.gpu.hip.HipDevice");
            GpuDevice hip = (GpuDevice) hipCls.getDeclaredConstructor().newInstance();
            if (hip.freeMemoryBytes() > 0) {
                return hip;
            }
            try {
                hip.close();
            } catch (Exception ignored) {
            }
        } catch (Exception ignored) {
        }

        try {
            Class<?> syclCls = Class.forName("com.github.tensor4j.gpu.sycl.SyclDevice");
            GpuDevice sycl = (GpuDevice) syclCls.getDeclaredConstructor().newInstance();
            if (sycl.freeMemoryBytes() > 0) {
                return sycl;
            }
            try {
                sycl.close();
            } catch (Exception ignored) {
            }
        } catch (Exception ignored) {
        }

        try {
            Class<?> oclCls = Class.forName("com.github.tensor4j.gpu.opencl.OclDevice");
            GpuDevice ocl = (GpuDevice) oclCls.getDeclaredConstructor().newInstance();
            if (ocl.freeMemoryBytes() > 0) {
                return ocl;
            }
            try {
                ocl.close();
            } catch (Exception ignored) {
            }
        } catch (Exception ignored) {
        }

        return new CpuDevice();
    }
}
