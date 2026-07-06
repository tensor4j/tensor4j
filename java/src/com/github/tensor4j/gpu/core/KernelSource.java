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

/**
 * Generates CUDA C and OpenCL C kernel source from a unified operation description.
 * This is the core of the "better bridge" — write once, target any backend.
 */
public final class KernelSource {

    private KernelSource() {
    }

    public static String addCuda() {
        return "extern \"C\" __global__ void add(float* a, float* b, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = a[i] + b[i]; }\n" +
               "}";
    }

    public static String addOpencl() {
        return "__kernel void add(__global float* a, __global float* b, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = a[i] + b[i]; }\n" +
               "}";
    }

    public static String subCuda() {
        return "extern \"C\" __global__ void sub(float* a, float* b, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = a[i] - b[i]; }\n" +
               "}";
    }

    public static String subOpencl() {
        return "__kernel void sub(__global float* a, __global float* b, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = a[i] - b[i]; }\n" +
               "}";
    }

    public static String mulCuda() {
        return "extern \"C\" __global__ void mul(float* a, float* b, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = a[i] * b[i]; }\n" +
               "}";
    }

    public static String mulOpencl() {
        return "__kernel void mul(__global float* a, __global float* b, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = a[i] * b[i]; }\n" +
               "}";
    }

    public static String divCuda() {
        return "extern \"C\" __global__ void div(float* a, float* b, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = a[i] / b[i]; }\n" +
               "}";
    }

    public static String divOpencl() {
        return "__kernel void div(__global float* a, __global float* b, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = a[i] / b[i]; }\n" +
               "}";
    }

    public static String reluCuda() {
        return "extern \"C\" __global__ void relu(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { float v = a[i]; c[i] = v > 0.0f ? v : 0.0f; }\n" +
               "}";
    }

    public static String reluOpencl() {
        return "__kernel void relu(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { float v = a[i]; c[i] = v > 0.0f ? v : 0.0f; }\n" +
               "}";
    }

    public static String negCuda() {
        return "extern \"C\" __global__ void neg(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = -a[i]; }\n" +
               "}";
    }

    public static String negOpencl() {
        return "__kernel void neg(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = -a[i]; }\n" +
               "}";
    }

    public static String scaleCuda() {
        return "extern \"C\" __global__ void scale(float* a, float s, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = a[i] * s; }\n" +
               "}";
    }

    public static String scaleOpencl() {
        return "__kernel void scale(__global float* a, float s, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = a[i] * s; }\n" +
               "}";
    }

    public static String expCuda() {
        return "extern \"C\" __global__ void exp(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = expf(a[i]); }\n" +
               "}";
    }

    public static String expOpencl() {
        return "__kernel void exp(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = exp(a[i]); }\n" +
               "}";
    }

    public static String logCuda() {
        return "extern \"C\" __global__ void log(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = logf(a[i]); }\n" +
               "}";
    }

    public static String logOpencl() {
        return "__kernel void log(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = log(a[i]); }\n" +
               "}";
    }

    public static String matmulCuda() {
        return "extern \"C\" __global__ void matmul(float* a, float* b, float* c, int m, int k, int n) {\n" +
               "    __shared__ float aTile[16][16];\n" +
               "    __shared__ float bTile[16][16];\n" +
               "    int row = blockIdx.y * 16 + threadIdx.y;\n" +
               "    int col = blockIdx.x * 16 + threadIdx.x;\n" +
               "    float sum = 0.0f;\n" +
               "    for (int t = 0; t < (k + 15) / 16; t++) {\n" +
               "        if (row < m && t * 16 + threadIdx.x < k)\n" +
               "            aTile[threadIdx.y][threadIdx.x] = a[row * k + t * 16 + threadIdx.x];\n" +
               "        else\n" +
               "            aTile[threadIdx.y][threadIdx.x] = 0.0f;\n" +
               "        if (col < n && t * 16 + threadIdx.y < k)\n" +
               "            bTile[threadIdx.y][threadIdx.x] = b[(t * 16 + threadIdx.y) * n + col];\n" +
               "        else\n" +
               "            bTile[threadIdx.y][threadIdx.x] = 0.0f;\n" +
               "        __syncthreads();\n" +
               "        for (int i = 0; i < 16; i++)\n" +
               "            sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];\n" +
               "        __syncthreads();\n" +
               "    }\n" +
               "    if (row < m && col < n)\n" +
               "        c[row * n + col] = sum;\n" +
               "}";
    }

    public static String matmulOpencl() {
        return "__kernel void matmul(__global float* a, __global float* b, __global float* c, int m, int k, int n) {\n" +
               "    __local float aTile[16][16];\n" +
               "    __local float bTile[16][16];\n" +
               "    int row = get_group_id(1) * 16 + get_local_id(1);\n" +
               "    int col = get_group_id(0) * 16 + get_local_id(0);\n" +
               "    float sum = 0.0f;\n" +
               "    for (int t = 0; t < (k + 15) / 16; t++) {\n" +
               "        if (row < m && t * 16 + get_local_id(0) < k)\n" +
               "            aTile[get_local_id(1)][get_local_id(0)] = a[row * k + t * 16 + get_local_id(0)];\n" +
               "        else\n" +
               "            aTile[get_local_id(1)][get_local_id(0)] = 0.0f;\n" +
               "        if (col < n && t * 16 + get_local_id(1) < k)\n" +
               "            bTile[get_local_id(1)][get_local_id(0)] = b[(t * 16 + get_local_id(1)) * n + col];\n" +
               "        else\n" +
               "            bTile[get_local_id(1)][get_local_id(0)] = 0.0f;\n" +
               "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "        for (int i = 0; i < 16; i++)\n" +
               "            sum += aTile[get_local_id(1)][i] * bTile[i][get_local_id(0)];\n" +
               "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    }\n" +
               "    if (row < m && col < n)\n" +
               "        c[row * n + col] = sum;\n" +
               "}";
    }

    public static String sqrtCuda() {
        return "extern \"C\" __global__ void sqrt(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = sqrtf(a[i]); }\n" +
               "}";
    }

    public static String sqrtOpencl() {
        return "__kernel void sqrt(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = sqrt(a[i]); }\n" +
               "}";
    }

    public static String sigmoidCuda() {
        return "extern \"C\" __global__ void sigmoid(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = 1.0f / (1.0f + expf(-a[i])); }\n" +
               "}";
    }

    public static String sigmoidOpencl() {
        return "__kernel void sigmoid(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = 1.0f / (1.0f + exp(-a[i])); }\n" +
               "}";
    }

    public static String tanhCuda() {
        return "extern \"C\" __global__ void tanh(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = tanhf(a[i]); }\n" +
               "}";
    }

    public static String tanhOpencl() {
        return "__kernel void tanh(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = tanh(a[i]); }\n" +
               "}";
    }

    public static String squaredDifferenceCuda() {
        return "extern \"C\" __global__ void squaredDifference(float* a, float* b, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { float d = a[i] - b[i]; c[i] = d * d; }\n" +
               "}";
    }

    public static String squaredDifferenceOpencl() {
        return "__kernel void squaredDifference(__global float* a, __global float* b, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { float d = a[i] - b[i]; c[i] = d * d; }\n" +
               "}";
    }

    public static String sumCuda() {
        return "extern \"C\" __global__ void sum(float* a, float* partial, int n) {\n" +
               "    __shared__ float cache[256];\n" +
               "    int tid = threadIdx.x;\n" +
               "    int i = blockIdx.x * blockDim.x + tid;\n" +
               "    float s = 0.0f;\n" +
               "    while (i < n) { s += a[i]; i += gridDim.x * blockDim.x; }\n" +
               "    cache[tid] = s;\n" +
               "    __syncthreads();\n" +
               "    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) { cache[tid] += cache[tid + sz]; }\n" +
               "        __syncthreads();\n" +
               "    }\n" +
               "    if (tid == 0) { partial[blockIdx.x] = cache[0]; }\n" +
               "}";
    }

    public static String sumOpencl() {
        return "__kernel void sum(__global float* a, __global float* partial, int n) {\n" +
               "    __local float cache[256];\n" +
               "    int tid = get_local_id(0);\n" +
               "    int i = get_global_id(0);\n" +
               "    float s = 0.0f;\n" +
               "    while (i < n) { s += a[i]; i += get_global_size(0); }\n" +
               "    cache[tid] = s;\n" +
               "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    for (int sz = get_local_size(0) / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) { cache[tid] += cache[tid + sz]; }\n" +
               "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    }\n" +
               "    if (tid == 0) { partial[get_group_id(0)] = cache[0]; }\n" +
               "}";
    }

    public static String maxCuda() {
        return "extern \"C\" __global__ void max(float* a, float* partial, int n) {\n" +
               "    __shared__ float cache[256];\n" +
               "    int tid = threadIdx.x;\n" +
               "    int i = blockIdx.x * blockDim.x + tid;\n" +
               "    float m = -__FLT_MAX__;\n" +
               "    while (i < n) { if (a[i] > m) m = a[i]; i += gridDim.x * blockDim.x; }\n" +
               "    cache[tid] = m;\n" +
               "    __syncthreads();\n" +
               "    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) { cache[tid] = fmaxf(cache[tid], cache[tid + sz]); }\n" +
               "        __syncthreads();\n" +
               "    }\n" +
               "    if (tid == 0) { partial[blockIdx.x] = cache[0]; }\n" +
               "}";
    }

    public static String maxOpencl() {
        return "__kernel void max(__global float* a, __global float* partial, int n) {\n" +
               "    __local float cache[256];\n" +
               "    int tid = get_local_id(0);\n" +
               "    int i = get_global_id(0);\n" +
               "    float m = -FLT_MAX;\n" +
               "    while (i < n) { if (a[i] > m) m = a[i]; i += get_global_size(0); }\n" +
               "    cache[tid] = m;\n" +
               "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    for (int sz = get_local_size(0) / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) { cache[tid] = fmax(cache[tid], cache[tid + sz]); }\n" +
               "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    }\n" +
               "    if (tid == 0) { partial[get_group_id(0)] = cache[0]; }\n" +
               "}";
    }

    public static String powCuda() {
        return "extern \"C\" __global__ void pow(float* a, float* b, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = powf(a[i], b[i]); }\n" +
               "}";
    }

    public static String powOpencl() {
        return "__kernel void pow(__global float* a, __global float* b, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = pow(a[i], b[i]); }\n" +
               "}";
    }

    public static String meanCuda() {
        return "extern \"C\" __global__ void mean(float* a, float* partial, int n) {\n" +
               "    __shared__ float cache[256];\n" +
               "    int tid = threadIdx.x;\n" +
               "    int i = blockIdx.x * blockDim.x + tid;\n" +
               "    float s = 0.0f;\n" +
               "    while (i < n) { s += a[i]; i += gridDim.x * blockDim.x; }\n" +
               "    cache[tid] = s;\n" +
               "    __syncthreads();\n" +
               "    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) { cache[tid] += cache[tid + sz]; }\n" +
               "        __syncthreads();\n" +
               "    }\n" +
               "    if (tid == 0) { partial[blockIdx.x] = cache[0]; }\n" +
               "}";
    }

    public static String meanOpencl() {
        return "__kernel void mean(__global float* a, __global float* partial, int n) {\n" +
               "    __local float cache[256];\n" +
               "    int tid = get_local_id(0);\n" +
               "    int i = get_global_id(0);\n" +
               "    float s = 0.0f;\n" +
               "    while (i < n) { s += a[i]; i += get_global_size(0); }\n" +
               "    cache[tid] = s;\n" +
               "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    for (int sz = get_local_size(0) / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) { cache[tid] += cache[tid + sz]; }\n" +
               "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    }\n" +
               "    if (tid == 0) { partial[get_group_id(0)] = cache[0]; }\n" +
               "}";
    }

    public static String absCuda() {
        return "extern \"C\" __global__ void abs(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = fabsf(a[i]); }\n" +
               "}";
    }

    public static String absOpencl() {
        return "__kernel void abs(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = fabs(a[i]); }\n" +
               "}";
    }

    public static String clipCuda() {
        return "extern \"C\" __global__ void clip(float* a, float minVal, float maxVal, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = fminf(fmaxf(a[i], minVal), maxVal); }\n" +
               "}";
    }

    public static String clipOpencl() {
        return "__kernel void clip(__global float* a, float minVal, float maxVal, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = fmin(fmax(a[i], minVal), maxVal); }\n" +
               "}";
    }

    public static String transposeCuda() {
        return "extern \"C\" __global__ void transpose(float* a, float* c, int rows, int cols) {\n" +
               "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n" +
               "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (row < rows && col < cols) {\n" +
               "        c[col * rows + row] = a[row * cols + col];\n" +
               "    }\n" +
               "}";
    }

    public static String transposeOpencl() {
        return "__kernel void transpose(__global float* a, __global float* c, int rows, int cols) {\n" +
               "    int row = get_global_id(1);\n" +
               "    int col = get_global_id(0);\n" +
               "    if (row < rows && col < cols) {\n" +
               "        c[col * rows + row] = a[row * cols + col];\n" +
               "    }\n" +
               "}";
    }

    public static String sinCuda() {
        return "extern \"C\" __global__ void sin(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = sinf(a[i]); }\n" +
               "}";
    }

    public static String sinOpencl() {
        return "__kernel void sin(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = sin(a[i]); }\n" +
               "}";
    }

    public static String cosCuda() {
        return "extern \"C\" __global__ void cos(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = cosf(a[i]); }\n" +
               "}";
    }

    public static String cosOpencl() {
        return "__kernel void cos(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = cos(a[i]); }\n" +
               "}";
    }

    public static String softmax2dCuda() {
        return "extern \"C\" __global__ void softmax2d(float* a, float* c, int rows, int cols) {\n" +
               "    __shared__ float cache[256];\n" +
               "    int row = blockIdx.x;\n" +
               "    int tid = threadIdx.x;\n" +
               "    if (row >= rows) return;\n" +
               "    int base = row * cols;\n" +
               "    float m = -__FLT_MAX__;\n" +
               "    for (int j = tid; j < cols; j += blockDim.x) {\n" +
               "        float v = a[base + j];\n" +
               "        if (v > m) m = v;\n" +
               "    }\n" +
               "    cache[tid] = m;\n" +
               "    __syncthreads();\n" +
               "    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) cache[tid] = fmaxf(cache[tid], cache[tid + sz]);\n" +
               "        __syncthreads();\n" +
               "    }\n" +
               "    float rowMax = cache[0];\n" +
               "    float s = 0.0f;\n" +
               "    for (int j = tid; j < cols; j += blockDim.x) {\n" +
               "        float e = expf(a[base + j] - rowMax);\n" +
               "        c[base + j] = e;\n" +
               "        s += e;\n" +
               "    }\n" +
               "    cache[tid] = s;\n" +
               "    __syncthreads();\n" +
               "    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) cache[tid] += cache[tid + sz];\n" +
               "        __syncthreads();\n" +
               "    }\n" +
               "    float rowSum = cache[0];\n" +
               "    for (int j = tid; j < cols; j += blockDim.x) {\n" +
               "        c[base + j] /= rowSum;\n" +
               "    }\n" +
               "}";
    }

    public static String softmax2dOpencl() {
        return "__kernel void softmax2d(__global float* a, __global float* c, int rows, int cols) {\n" +
               "    __local float cache[256];\n" +
               "    int row = get_group_id(0);\n" +
               "    int tid = get_local_id(0);\n" +
               "    if (row >= rows) return;\n" +
               "    int base = row * cols;\n" +
               "    float m = -FLT_MAX;\n" +
               "    for (int j = tid; j < cols; j += get_local_size(0)) {\n" +
               "        float v = a[base + j];\n" +
               "        if (v > m) m = v;\n" +
               "    }\n" +
               "    cache[tid] = m;\n" +
               "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    for (int sz = get_local_size(0) / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) cache[tid] = fmax(cache[tid], cache[tid + sz]);\n" +
               "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    }\n" +
               "    float rowMax = cache[0];\n" +
               "    float s = 0.0f;\n" +
               "    for (int j = tid; j < cols; j += get_local_size(0)) {\n" +
               "        float e = exp(a[base + j] - rowMax);\n" +
               "        c[base + j] = e;\n" +
               "        s += e;\n" +
               "    }\n" +
               "    cache[tid] = s;\n" +
               "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    for (int sz = get_local_size(0) / 2; sz > 0; sz >>= 1) {\n" +
               "        if (tid < sz) cache[tid] += cache[tid + sz];\n" +
               "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
               "    }\n" +
               "    float rowSum = cache[0];\n" +
               "    for (int j = tid; j < cols; j += get_local_size(0)) {\n" +
               "        c[base + j] /= rowSum;\n" +
               "    }\n" +
               "}";
    }

    public static String siluCuda() {
        return "extern \"C\" __global__ void silu(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { float x = a[i]; c[i] = x / (1.0f + expf(-x)); }\n" +
               "}";
    }

    public static String siluOpencl() {
        return "__kernel void silu(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { float x = a[i]; c[i] = x / (1.0f + exp(-x)); }\n" +
               "}";
    }

    public static String eluCuda() {
        return "extern \"C\" __global__ void elu(float* a, float alpha, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { float x = a[i]; c[i] = x > 0.0f ? x : alpha * (expf(x) - 1.0f); }\n" +
               "}";
    }

    public static String eluOpencl() {
        return "__kernel void elu(__global float* a, float alpha, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { float x = a[i]; c[i] = x > 0.0f ? x : alpha * (exp(x) - 1.0f); }\n" +
               "}";
    }

    public static String subScalarCuda() {
        return "extern \"C\" __global__ void subScalar(float* a, float s, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = a[i] - s; }\n" +
               "}";
    }

    public static String subScalarOpencl() {
        return "__kernel void subScalar(__global float* a, float s, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = a[i] - s; }\n" +
               "}";
    }

    public static String binaryCrossEntropyCuda() {
        return "extern \"C\" __global__ void binaryCrossEntropy(float* pred, float* target, float* loss, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) {\n" +
               "        float p = fmaxf(fminf(pred[i], 0.9999999f), 1e-7f);\n" +
               "        loss[i] = -target[i] * logf(p) - (1.0f - target[i]) * logf(1.0f - p);\n" +
               "    }\n" +
               "}";
    }

    public static String binaryCrossEntropyOpencl() {
        return "__kernel void binaryCrossEntropy(__global float* pred, __global float* target, __global float* loss, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) {\n" +
               "        float p = fmax(fmin(pred[i], 0.9999999f), 1e-7f);\n" +
               "        loss[i] = -target[i] * log(p) - (1.0f - target[i]) * log(1.0f - p);\n" +
               "    }\n" +
               "}";
    }

    public static String dropoutCuda() {
        return "extern \"C\" __global__ void dropout(float* a, float* mask, float* c, float prob, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { c[i] = mask[i] < prob ? 0.0f : a[i] / (1.0f - prob); }\n" +
               "}";
    }

    public static String dropoutOpencl() {
        return "__kernel void dropout(__global float* a, __global float* mask, __global float* c, float prob, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { c[i] = mask[i] < prob ? 0.0f : a[i] / (1.0f - prob); }\n" +
               "}";
    }

    public static String geluCuda() {
        return "extern \"C\" __global__ void gelu(float* a, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { float x = a[i]; c[i] = 0.5f * x * (1.0f + erff(x * 0.70710678f)); }\n" +
               "}";
    }

    public static String geluOpencl() {
        return "__kernel void gelu(__global float* a, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { float x = a[i]; c[i] = 0.5f * x * (1.0f + erf(x * 0.70710678f)); }\n" +
               "}";
    }

    public static String leakyReluCuda() {
        return "extern \"C\" __global__ void leakyRelu(float* a, float alpha, float* c, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) { float v = a[i]; c[i] = v > 0.0f ? v : v * alpha; }\n" +
               "}";
    }

    public static String leakyReluOpencl() {
        return "__kernel void leakyRelu(__global float* a, float alpha, __global float* c, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) { float v = a[i]; c[i] = v > 0.0f ? v : v * alpha; }\n" +
               "}";
    }

    public static String expandCuda() {
        return "extern \"C\" __global__ void expand(float* a, int* aShape, int* aStrides, int aNd, float* c, int* cShape, int cNd, int n) {\n" +
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
               "    if (i < n) {\n" +
               "        int idx = i;\n" +
               "        int aIdx = 0;\n" +
               "        for (int d = cNd - 1; d >= 0; d--) {\n" +
               "            int cSize = cShape[d];\n" +
               "            int pos = idx % cSize;\n" +
               "            idx /= cSize;\n" +
               "            if (d < aNd) {\n" +
               "                int aSize = aShape[aNd - 1 - (cNd - 1 - d)];\n" +
               "                if (aSize > 1) { aIdx += pos * aStrides[aNd - 1 - (cNd - 1 - d)]; }\n" +
               "            }\n" +
               "        }\n" +
               "        c[i] = a[aIdx];\n" +
               "    }\n" +
               "}";
    }

    public static String expandOpencl() {
        return "__kernel void expand(__global float* a, __global int* aShape, __global int* aStrides, int aNd, __global float* c, __global int* cShape, int cNd, int n) {\n" +
               "    int i = get_global_id(0);\n" +
               "    if (i < n) {\n" +
               "        int idx = i;\n" +
               "        int aIdx = 0;\n" +
               "        for (int d = cNd - 1; d >= 0; d--) {\n" +
               "            int cSize = cShape[d];\n" +
               "            int pos = idx % cSize;\n" +
               "            idx /= cSize;\n" +
               "            if (d < aNd) {\n" +
               "                int aSize = aShape[aNd - 1 - (cNd - 1 - d)];\n" +
               "                if (aSize > 1) { aIdx += pos * aStrides[aNd - 1 - (cNd - 1 - d)]; }\n" +
               "            }\n" +
               "        }\n" +
               "        c[i] = a[aIdx];\n" +
               "    }\n" +
               "}";
    }
}
