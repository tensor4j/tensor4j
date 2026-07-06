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
 * Tensor math operations dispatching to GPU kernels.
 * All operations return new tensors; kernel source generated at runtime.
 * CPU fallback for CI/development without GPU hardware.
 */
public final class GpuTensorMath {

    private final GpuDevice device;
    private final boolean isCpuRef;
    private GpuProgram progAdd;
    private GpuProgram progSub;
    private GpuProgram progMul;
    private GpuProgram progDiv;
    private GpuProgram progRelu;
    private GpuProgram progNeg;
    private GpuProgram progScale;
    private GpuProgram progExp;
    private GpuProgram progLog;
    private GpuProgram progMatmul;
    private GpuProgram progSqrt;
    private GpuProgram progSigmoid;
    private GpuProgram progTanh;
    private GpuProgram progSquaredDifference;
    private GpuProgram progSum;
    private GpuProgram progMax;
    private GpuProgram progPow;
    private GpuProgram progMean;
    private GpuProgram progAbs;
    private GpuProgram progClip;
    private GpuProgram progTranspose;
    private GpuProgram progSin;
    private GpuProgram progCos;
    private GpuProgram progGelu;
    private GpuProgram progLeakyRelu;
    private GpuProgram progExpand;
    private GpuProgram progSoftmax2d;
    private GpuProgram progSubScalar;
    private GpuProgram progBinaryCrossEntropy;
    private GpuProgram progDropout;
    private GpuProgram progSilu;
    private GpuProgram progElu;

    public GpuTensorMath(GpuDevice device) {
        this.device = device;
        this.isCpuRef = device instanceof CpuDevice;
    }

    public GpuTensor add(GpuTensor a, GpuTensor b) {
        assertSameShape(a, b);
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuBinary(r, a, b, (x, y) -> x + y);
            return r;
        }
        if (progAdd == null) {
            progAdd = device.compile(kernelSourceFor("add"), "add");
        }
        launch1D(r.numel(), progAdd, "add",
                 bufs(a, b, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor sub(GpuTensor a, GpuTensor b) {
        assertSameShape(a, b);
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuBinary(r, a, b, (x, y) -> x - y);
            return r;
        }
        if (progSub == null) {
            progSub = device.compile(kernelSourceFor("sub"), "sub");
        }
        launch1D(r.numel(), progSub, "sub",
                 bufs(a, b, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor mul(GpuTensor a, GpuTensor b) {
        assertSameShape(a, b);
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuBinary(r, a, b, (x, y) -> x * y);
            return r;
        }
        if (progMul == null) {
            progMul = device.compile(kernelSourceFor("mul"), "mul");
        }
        launch1D(r.numel(), progMul, "mul",
                 bufs(a, b, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor div(GpuTensor a, GpuTensor b) {
        assertSameShape(a, b);
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuBinary(r, a, b, (x, y) -> x / y);
            return r;
        }
        if (progDiv == null) {
            progDiv = device.compile(kernelSourceFor("div"), "div");
        }
        launch1D(r.numel(), progDiv, "div",
                 bufs(a, b, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor relu(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                hc[i] = ha[i] > 0f ? ha[i] : 0f;
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progRelu == null) {
            progRelu = device.compile(kernelSourceFor("relu"), "relu");
        }
        launch1D(r.numel(), progRelu, "relu",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor neg(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                hc[i] = -ha[i];
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progNeg == null) {
            progNeg = device.compile(kernelSourceFor("neg"), "neg");
        }
        launch1D(r.numel(), progNeg, "neg",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor scale(GpuTensor a, float s) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                hc[i] = ha[i] * s;
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progScale == null) {
            progScale = device.compile(kernelSourceFor("scale"), "scale");
        }
        launch1D(r.numel(), progScale, "scale",
                 bufs(a, r), ints(r.numel()), floats(s));
        return r;
    }

    public GpuTensor exp(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                hc[i] = (float) Math.exp(ha[i]);
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progExp == null) {
            progExp = device.compile(kernelSourceFor("exp"), "exp");
        }
        launch1D(r.numel(), progExp, "exp",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor log(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                hc[i] = (float) Math.log(ha[i]);
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progLog == null) {
            progLog = device.compile(kernelSourceFor("log"), "log");
        }
        launch1D(r.numel(), progLog, "log",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor matmul(GpuTensor a, GpuTensor b) {
        int[] sa = a.shape();
        int[] sb = b.shape();
        if (sa.length != 2 || sb.length != 2) {
            throw new IllegalArgumentException("matmul requires 2D tensors");
        }
        int m = sa[0];
        int k = sa[1];
        int n = sb[1];
        if (sb[0] != k) {
            throw new IllegalArgumentException("matmul inner dim mismatch: " + k + " vs " + sb[0]);
        }
        GpuTensor r = allocate(new int[]{m, n});
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hb = b.toHost();
            float[] hc = new float[m * n];
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < n; col++) {
                    float sum = 0f;
                    for (int i = 0; i < k; i++) {
                        sum += ha[row * k + i] * hb[i * n + col];
                    }
                    hc[row * n + col] = sum;
                }
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progMatmul == null) {
            progMatmul = device.compile(kernelSourceFor("matmul"), "matmul");
        }
        launch2D(m, n, progMatmul, "matmul",
                 bufs(a, b, r),
                 ints(m, k, n),
                 floats());
        return r;
    }

    public GpuTensor sqrt(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> (float) Math.sqrt(x));
            return r;
        }
        if (progSqrt == null) {
            progSqrt = device.compile(kernelSourceFor("sqrt"), "sqrt");
        }
        launch1D(r.numel(), progSqrt, "sqrt",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor sigmoid(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> 1f / (1f + (float) Math.exp(-x)));
            return r;
        }
        if (progSigmoid == null) {
            progSigmoid = device.compile(kernelSourceFor("sigmoid"), "sigmoid");
        }
        launch1D(r.numel(), progSigmoid, "sigmoid",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor tanh(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> (float) Math.tanh(x));
            return r;
        }
        if (progTanh == null) {
            progTanh = device.compile(kernelSourceFor("tanh"), "tanh");
        }
        launch1D(r.numel(), progTanh, "tanh",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor squaredDifference(GpuTensor a, GpuTensor b) {
        assertSameShape(a, b);
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuBinary(r, a, b, (x, y) -> { float d = x - y; return d * d; });
            return r;
        }
        if (progSquaredDifference == null) {
            progSquaredDifference = device.compile(kernelSourceFor("squaredDifference"), "squaredDifference");
        }
        launch1D(r.numel(), progSquaredDifference, "squaredDifference",
                 bufs(a, b, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor sum(GpuTensor a) {
        if (isCpuRef) {
            float[] ha = a.toHost();
            float s = 0f;
            for (float v : ha) {
                s += v;
            }
            GpuTensor r = allocate(new int[]{1});
            r.buffer().copyFromHost(new float[]{s}, 0, 0, 1);
            return r;
        }
        int n = a.numel();
        int threads = 256;
        int blocks = Math.min((n + threads - 1) / threads, 1024);
        GpuBuffer partialBuf = device.allocate((long) blocks * 4L);
        GpuBuffer inputBuf = a.buffer();
        if (progSum == null) {
            progSum = device.compile(kernelSourceFor("sum"), "sum");
        }
        GpuKernel kernel = progSum.createKernel("sum");
        kernel.launch(device.createStream(), blocks, 1, 1, threads, 1, 1,
                      new GpuBuffer[]{inputBuf, partialBuf},
                      ints(n), floats());
        float[] partials = new float[blocks];
        partialBuf.copyToHost(partials, 0, 0, blocks);
        partialBuf.close();
        float total = 0f;
        for (float p : partials) {
            total += p;
        }
        GpuTensor r = allocate(new int[]{1});
        r.buffer().copyFromHost(new float[]{total}, 0, 0, 1);
        return r;
    }

    public GpuTensor max(GpuTensor a) {
        if (isCpuRef) {
            float[] ha = a.toHost();
            float m = Float.NEGATIVE_INFINITY;
            for (float v : ha) {
                if (v > m) m = v;
            }
            GpuTensor r = allocate(new int[]{1});
            r.buffer().copyFromHost(new float[]{m}, 0, 0, 1);
            return r;
        }
        int n = a.numel();
        int threads = 256;
        int blocks = Math.min((n + threads - 1) / threads, 1024);
        GpuBuffer partialBuf = device.allocate((long) blocks * 4L);
        GpuBuffer inputBuf = a.buffer();
        if (progMax == null) {
            progMax = device.compile(kernelSourceFor("max"), "max");
        }
        GpuKernel kernel = progMax.createKernel("max");
        kernel.launch(device.createStream(), blocks, 1, 1, threads, 1, 1,
                      new GpuBuffer[]{inputBuf, partialBuf},
                      ints(n), floats());
        float[] partials = new float[blocks];
        partialBuf.copyToHost(partials, 0, 0, blocks);
        partialBuf.close();
        float m = Float.NEGATIVE_INFINITY;
        for (float p : partials) {
            if (p > m) m = p;
        }
        GpuTensor r = allocate(new int[]{1});
        r.buffer().copyFromHost(new float[]{m}, 0, 0, 1);
        return r;
    }

    public GpuTensor pow(GpuTensor a, GpuTensor b) {
        assertSameShape(a, b);
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuBinary(r, a, b, (x, y) -> (float) Math.pow(x, y));
            return r;
        }
        if (progPow == null) {
            progPow = device.compile(kernelSourceFor("pow"), "pow");
        }
        launch1D(r.numel(), progPow, "pow",
                 bufs(a, b, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor mean(GpuTensor a) {
        if (isCpuRef) {
            float[] ha = a.toHost();
            float s = 0f;
            for (float v : ha) {
                s += v;
            }
            GpuTensor r = allocate(new int[]{1});
            r.buffer().copyFromHost(new float[]{s / ha.length}, 0, 0, 1);
            return r;
        }
        int n = a.numel();
        int threads = 256;
        int blocks = Math.min((n + threads - 1) / threads, 1024);
        GpuBuffer partialBuf = device.allocate((long) blocks * 4L);
        GpuBuffer inputBuf = a.buffer();
        if (progMean == null) {
            progMean = device.compile(kernelSourceFor("mean"), "mean");
        }
        GpuKernel kernel = progMean.createKernel("mean");
        kernel.launch(device.createStream(), blocks, 1, 1, threads, 1, 1,
                      new GpuBuffer[]{inputBuf, partialBuf},
                      ints(n), floats());
        float[] partials = new float[blocks];
        partialBuf.copyToHost(partials, 0, 0, blocks);
        partialBuf.close();
        float total = 0f;
        for (float p : partials) {
            total += p;
        }
        GpuTensor r = allocate(new int[]{1});
        r.buffer().copyFromHost(new float[]{total / n}, 0, 0, 1);
        return r;
    }

    public GpuTensor abs(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> Math.abs(x));
            return r;
        }
        if (progAbs == null) {
            progAbs = device.compile(kernelSourceFor("abs"), "abs");
        }
        launch1D(r.numel(), progAbs, "abs",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor clip(GpuTensor a, float minVal, float maxVal) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> Math.max(minVal, Math.min(maxVal, x)));
            return r;
        }
        if (progClip == null) {
            progClip = device.compile(kernelSourceFor("clip"), "clip");
        }
        launch1D(r.numel(), progClip, "clip",
                 bufs(a, r), ints(r.numel()), floats(minVal, maxVal));
        return r;
    }

    public GpuTensor transpose(GpuTensor a) {
        if (a.shape().length != 2) {
            throw new IllegalArgumentException("transpose requires 2D tensor");
        }
        int rows = a.shape()[0];
        int cols = a.shape()[1];
        GpuTensor r = allocate(new int[]{cols, rows});
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    hc[col * rows + row] = ha[row * cols + col];
                }
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progTranspose == null) {
            progTranspose = device.compile(kernelSourceFor("transpose"), "transpose");
        }
        launch2D(rows, cols, progTranspose, "transpose",
                 bufs(a, r),
                 ints(rows, cols),
                 floats());
        return r;
    }

    public GpuTensor sin(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> (float) Math.sin(x));
            return r;
        }
        if (progSin == null) {
            progSin = device.compile(kernelSourceFor("sin"), "sin");
        }
        launch1D(r.numel(), progSin, "sin",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor cos(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> (float) Math.cos(x));
            return r;
        }
        if (progCos == null) {
            progCos = device.compile(kernelSourceFor("cos"), "cos");
        }
        launch1D(r.numel(), progCos, "cos",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor gelu(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                float x = ha[i];
                hc[i] = 0.5f * x * (1f + (float) Math.tanh(
                    0.7978845608f * (x + 0.044715f * x * x * x)));
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progGelu == null) {
            progGelu = device.compile(kernelSourceFor("gelu"), "gelu");
        }
        launch1D(r.numel(), progGelu, "gelu",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor leakyRelu(GpuTensor a, float alpha) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                hc[i] = ha[i] > 0f ? ha[i] : ha[i] * alpha;
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progLeakyRelu == null) {
            progLeakyRelu = device.compile(kernelSourceFor("leakyRelu"), "leakyRelu");
        }
        launch1D(r.numel(), progLeakyRelu, "leakyRelu",
                 bufs(a, r), ints(r.numel()), floats(alpha));
        return r;
    }

    public GpuTensor silu(GpuTensor a) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> (float)(x / (1.0 + Math.exp(-x))));
            return r;
        }
        if (progSilu == null) {
            progSilu = device.compile(kernelSourceFor("silu"), "silu");
        }
        launch1D(r.numel(), progSilu, "silu",
                 bufs(a, r), ints(r.numel()), floats());
        return r;
    }

    public GpuTensor elu(GpuTensor a, float alpha) {
        GpuTensor r = allocate(a.shape());
        if (isCpuRef) {
            cpuUnary(r, a, x -> x > 0f ? x : alpha * ((float)Math.exp(x) - 1f));
            return r;
        }
        if (progElu == null) {
            progElu = device.compile(kernelSourceFor("elu"), "elu");
        }
        launch1D(r.numel(), progElu, "elu",
                 bufs(a, r), ints(r.numel()), floats(alpha));
        return r;
    }

    public GpuTensor expand(GpuTensor a, int... targetShape) {
        int targetNumel = 1;
        for (int d : targetShape) {
            targetNumel *= d;
        }
        GpuTensor r = allocate(targetShape);
        if (isCpuRef) {
            float[] ha = a.toHost();
            int[] aShape = a.shape();
            int[] aStrides = a.strides();
            float[] hc = new float[targetNumel];
            int[] cShape = targetShape;
            int aNd = aShape.length;
            int cNd = targetShape.length;
            for (int flat = 0; flat < targetNumel; flat++) {
                int idx = flat;
                int aIdx = 0;
                for (int d = cNd - 1; d >= 0; d--) {
                    int cSize = cShape[d];
                    int pos = idx % cSize;
                    idx /= cSize;
                    if (d >= cNd - aNd) {
                        int aDim = d - (cNd - aNd);
                        if (aShape[aDim] > 1) {
                            aIdx += pos * aStrides[aDim];
                        }
                    }
                }
                hc[flat] = ha[aIdx];
            }
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        if (progExpand == null) {
            progExpand = device.compile(kernelSourceFor("expand"), "expand");
        }
        GpuBuffer shapeBuf = device.allocate((long) a.shape().length * 4L);
        shapeBuf.copyFromHost(toFloatArray(a.shape()), 0, 0, a.shape().length);
        GpuBuffer stridesBuf = device.allocate((long) a.strides().length * 4L);
        stridesBuf.copyFromHost(toFloatArray(a.strides()), 0, 0, a.strides().length);
        GpuBuffer targetShapeBuf = device.allocate((long) targetShape.length * 4L);
        targetShapeBuf.copyFromHost(toFloatArray(targetShape), 0, 0, targetShape.length);
        launch1D(targetNumel, progExpand, "expand",
                 bufs(a,
                     new GpuTensor(shapeBuf, new int[]{a.shape().length}, GpuTensor.computeStrides(new int[]{a.shape().length}), 0),
                     new GpuTensor(stridesBuf, new int[]{a.strides().length}, GpuTensor.computeStrides(new int[]{a.strides().length}), 0),
                     r,
                     new GpuTensor(targetShapeBuf, new int[]{targetShape.length}, GpuTensor.computeStrides(new int[]{targetShape.length}), 0)),
                 ints(a.shape().length, targetShape.length, targetNumel),
                 floats());
        return r;
    }

    private static float[] toFloatArray(int[] arr) {
        float[] f = new float[arr.length];
        for (int i = 0; i < arr.length; i++) f[i] = (float) arr[i];
        return f;
    }

    public GpuTensor softmax(GpuTensor a) {
        int[] shape = a.shape();
        if (shape.length == 1) {
            return softmax1d(a);
        } else if (shape.length == 2) {
            return softmax2d(a);
        }
        throw new IllegalArgumentException("softmax requires 1D or 2D tensor");
    }

    private GpuTensor softmax1d(GpuTensor a) {
        if (isCpuRef) {
            float[] ha = a.toHost();
            float max = Float.NEGATIVE_INFINITY;
            for (float v : ha) {
                if (v > max) max = v;
            }
            float sum = 0f;
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                hc[i] = (float) Math.exp(ha[i] - max);
                sum += hc[i];
            }
            for (int i = 0; i < ha.length; i++) {
                hc[i] /= sum;
            }
            GpuTensor r = allocate(a.shape());
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        GpuTensor maxT = max(a);
        float maxVal = maxT.toHost()[0];
        maxT.close();
        int n = a.numel();
        GpuTensor shifted = allocate(a.shape());
        {
            if (progSubScalar == null) {
                progSubScalar = device.compile(kernelSourceFor("subScalar"), "subScalar");
            }
            GpuKernel subScalarKernel = progSubScalar.createKernel("subScalar");
            subScalarKernel.launch(device.createStream(), 1, 1, 1, 256, 1, 1,
                                   new GpuBuffer[]{a.buffer(), shifted.buffer()},
                                   ints(n), floats(maxVal));
        }
        if (progExp == null) {
            progExp = device.compile(kernelSourceFor("exp"), "exp");
        }
        GpuKernel expKernel = progExp.createKernel("exp");
        expKernel.launch(device.createStream(), 1, 1, 1, 256, 1, 1,
                         new GpuBuffer[]{shifted.buffer(), shifted.buffer()},
                         ints(n), floats());
        GpuTensor sumT = sum(shifted);
        float sumVal = sumT.toHost()[0];
        sumT.close();
        GpuTensor r = allocate(a.shape());
        {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            if (progScale == null) {
                progScale = device.compile(kernelSourceFor("scale"), "scale");
            }
            GpuKernel scaleKernel = progScale.createKernel("scale");
            scaleKernel.launch(device.createStream(), blocks, 1, 1, threads, 1, 1,
                               new GpuBuffer[]{shifted.buffer(), r.buffer()},
                               ints(n), floats(1f / sumVal));
        }
        shifted.close();
        return r;
    }

    GpuTensor softmax2d(GpuTensor a) {
        int[] shape = a.shape();
        int rows = shape[0];
        int cols = shape[1];
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[rows * cols];
            for (int r = 0; r < rows; r++) {
                int base = r * cols;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < cols; j++) {
                    if (ha[base + j] > max) max = ha[base + j];
                }
                float sum = 0f;
                for (int j = 0; j < cols; j++) {
                    hc[base + j] = (float) Math.exp(ha[base + j] - max);
                    sum += hc[base + j];
                }
                for (int j = 0; j < cols; j++) {
                    hc[base + j] /= sum;
                }
            }
            GpuTensor r = allocate(shape);
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        GpuTensor r = allocate(shape);
        if (progSoftmax2d == null) {
            progSoftmax2d = device.compile(kernelSourceFor("softmax2d"), "softmax2d");
        }
        int threads = Math.min(cols, 256);
        GpuKernel kernel = progSoftmax2d.createKernel("softmax2d");
        kernel.launch(device.createStream(), rows, 1, 1, threads, 1, 1,
                       new GpuBuffer[]{a.buffer(), r.buffer()},
                       ints(rows, cols), floats());
        return r;
    }

    public GpuTensor dropout(GpuTensor a, float prob, java.util.Random rng) {
        prob = Math.max(0f, Math.min(1f, prob));
        if (prob == 0f) {
            GpuTensor r = allocate(a.shape());
            r.buffer().copyFromHost(a.toHost(), 0, 0, a.numel());
            return r;
        }
        if (isCpuRef) {
            float[] ha = a.toHost();
            float[] hc = new float[ha.length];
            for (int i = 0; i < ha.length; i++) {
                hc[i] = rng.nextFloat() < prob ? 0f : ha[i] / (1f - prob);
            }
            GpuTensor r = allocate(a.shape());
            r.buffer().copyFromHost(hc, 0, 0, hc.length);
            return r;
        }
        int n = a.numel();
        float[] mask = new float[n];
        for (int i = 0; i < n; i++) {
            mask[i] = rng.nextFloat();
        }
        GpuTensor maskT = allocate(a.shape());
        maskT.buffer().copyFromHost(mask, 0, 0, n);
        GpuTensor r = allocate(a.shape());
        if (progDropout == null) {
            progDropout = device.compile(kernelSourceFor("dropout"), "dropout");
        }
        launch1D(n, progDropout, "dropout",
                 bufs(a, maskT, r), ints(n), floats(prob));
        maskT.close();
        return r;
    }

    public GpuTensor binaryCrossEntropy(GpuTensor pred, GpuTensor target) {
        assertSameShape(pred, target);
        int n = pred.numel();
        if (isCpuRef) {
            float[] hp = pred.toHost();
            float[] ht = target.toHost();
            float sum = 0f;
            for (int i = 0; i < n; i++) {
                float p = Math.max(Math.min(hp[i], 0.9999999f), 1e-7f);
                sum += -ht[i] * Math.log(p) - (1f - ht[i]) * Math.log(1f - p);
            }
            GpuTensor r = allocate(new int[]{1});
            r.buffer().copyFromHost(new float[]{sum}, 0, 0, 1);
            return r;
        }
        GpuTensor loss = allocate(new int[]{n});
        if (progBinaryCrossEntropy == null) {
            progBinaryCrossEntropy = device.compile(kernelSourceFor("binaryCrossEntropy"), "binaryCrossEntropy");
        }
        launch1D(n, progBinaryCrossEntropy, "binaryCrossEntropy",
                 bufs(pred, target, loss), ints(n), floats());
        float[] hLoss = loss.toHost();
        float sum = 0f;
        for (int i = 0; i < n; i++) sum += hLoss[i];
        loss.close();
        GpuTensor r = allocate(new int[]{1});
        r.buffer().copyFromHost(new float[]{sum}, 0, 0, 1);
        return r;
    }

    public GpuTensor crossEntropy(GpuTensor logits, GpuTensor targets) {
        int[] ls = logits.shape();
        if (ls.length != 2) {
            throw new IllegalArgumentException("crossEntropy requires 2D logits [N, C]");
        }
        int n = ls[0];
        int c = ls[1];
        GpuTensor probs = softmax2d(logits);
        float[] hp = probs.toHost();
        float[] ht = targets.toHost();
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            int t = (int) ht[i];
            if (t < 0 || t >= c) {
                throw new IllegalArgumentException("target index out of range: " + t);
            }
            float p = Math.max(hp[i * c + t], 1e-7f);
            sum += -Math.log(p);
        }
        GpuTensor r = allocate(new int[]{1});
        r.buffer().copyFromHost(new float[]{sum}, 0, 0, 1);
        return r;
    }

    public GpuTensor layerNorm(GpuTensor x, GpuTensor gamma, GpuTensor beta, float eps) {
        int[] xs = x.shape();
        if (xs.length != 2) {
            throw new IllegalArgumentException("layerNorm requires 2D input [B, D]");
        }
        int b = xs[0];
        int d = xs[1];
        float[] hx = x.toHost();
        float[] hg = gamma.toHost();
        float[] hb = beta.toHost();
        float[] hc = new float[b * d];
        for (int i = 0; i < b; i++) {
            int base = i * d;
            float mean = 0f;
            for (int j = 0; j < d; j++) mean += hx[base + j];
            mean /= d;
            float var = 0f;
            for (int j = 0; j < d; j++) {
                float diff = hx[base + j] - mean;
                var += diff * diff;
            }
            var /= d;
            float invStd = 1f / (float) Math.sqrt(var + eps);
            for (int j = 0; j < d; j++) {
                hc[base + j] = hg[j] * (hx[base + j] - mean) * invStd + hb[j];
            }
        }
        GpuTensor r = allocate(xs);
        r.buffer().copyFromHost(hc, 0, 0, hc.length);
        return r;
    }

    public GpuTensor batchNorm1d(GpuTensor x, GpuTensor gamma, GpuTensor beta, float eps) {
        int[] xs = x.shape();
        if (xs.length != 2) {
            throw new IllegalArgumentException("batchNorm1d requires 2D input [B, D]");
        }
        int b = xs[0];
        int d = xs[1];
        float[] hx = x.toHost();
        float[] hg = gamma.toHost();
        float[] hb = beta.toHost();
        float[] hc = new float[b * d];
        for (int j = 0; j < d; j++) {
            float mean = 0f;
            for (int i = 0; i < b; i++) mean += hx[i * d + j];
            mean /= b;
            float var = 0f;
            for (int i = 0; i < b; i++) {
                float diff = hx[i * d + j] - mean;
                var += diff * diff;
            }
            var /= b;
            float invStd = 1f / (float) Math.sqrt(var + eps);
            for (int i = 0; i < b; i++) {
                hc[i * d + j] = hg[j] * (hx[i * d + j] - mean) * invStd + hb[j];
            }
        }
        GpuTensor r = allocate(xs);
        r.buffer().copyFromHost(hc, 0, 0, hc.length);
        return r;
    }

    public GpuTensor flatten(GpuTensor a, int startDim) {
        int[] shape = a.shape();
        if (startDim < 0 || startDim >= shape.length) {
            throw new IllegalArgumentException("startDim out of range: " + startDim);
        }
        int outer = 1;
        for (int i = 0; i < startDim; i++) outer *= shape[i];
        int inner = 1;
        for (int i = startDim; i < shape.length; i++) inner *= shape[i];
        int[] newShape = new int[startDim + 1];
        for (int i = 0; i < startDim; i++) newShape[i] = shape[i];
        newShape[startDim] = inner;
        if (startDim == 0) {
            newShape = new int[]{inner};
        }
        float[] ha = a.toHost();
        GpuTensor r = allocate(newShape);
        r.buffer().copyFromHost(ha, 0, 0, ha.length);
        return r;
    }

    public GpuTensor argmax(GpuTensor a, int dim) {
        int[] shape = a.shape();
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("dim out of range: " + dim);
        }
        float[] ha = a.toHost();
        if (shape.length == 1) {
            int n = shape[0];
            int bestIdx = 0;
            float bestVal = ha[0];
            for (int i = 1; i < n; i++) {
                if (ha[i] > bestVal) { bestVal = ha[i]; bestIdx = i; }
            }
            GpuTensor r = allocate(new int[]{1});
            r.buffer().copyFromHost(new float[]{bestIdx}, 0, 0, 1);
            return r;
        }
        if (shape.length == 2 && dim == 1) {
            int rows = shape[0];
            int cols = shape[1];
            float[] hc = new float[rows];
            for (int row = 0; row < rows; row++) {
                int bestIdx = 0;
                float bestVal = ha[row * cols];
                for (int j = 1; j < cols; j++) {
                    if (ha[row * cols + j] > bestVal) { bestVal = ha[row * cols + j]; bestIdx = j; }
                }
                hc[row] = bestIdx;
            }
            GpuTensor r = allocate(new int[]{rows});
            r.buffer().copyFromHost(hc, 0, 0, rows);
            return r;
        }
        throw new UnsupportedOperationException("argmax only supports 1D or 2D dim=1");
    }

    public GpuTensor oneHot(GpuTensor indices, int numClasses) {
        float[] hi = indices.toHost();
        int n = hi.length;
        float[] hc = new float[n * numClasses];
        for (int i = 0; i < n; i++) {
            int idx = (int) hi[i];
            if (idx < 0 || idx >= numClasses) {
                throw new IllegalArgumentException("index out of range: " + idx);
            }
            hc[i * numClasses + idx] = 1f;
        }
        GpuTensor r = allocate(new int[]{n, numClasses});
        r.buffer().copyFromHost(hc, 0, 0, hc.length);
        return r;
    }

    public GpuTensor embedding(GpuTensor weight, GpuTensor indices) {
        float[] hw = weight.toHost();
        float[] hi = indices.toHost();
        int[] ws = weight.shape();
        int numEmbed = ws[0];
        int dim = ws[1];
        int n = hi.length;
        float[] hc = new float[n * dim];
        for (int i = 0; i < n; i++) {
            int idx = (int) hi[i];
            if (idx < 0 || idx >= numEmbed) {
                throw new IllegalArgumentException("embedding index out of range: " + idx);
            }
            System.arraycopy(hw, idx * dim, hc, i * dim, dim);
        }
        GpuTensor r = allocate(new int[]{n, dim});
        r.buffer().copyFromHost(hc, 0, 0, hc.length);
        return r;
    }

    public GpuTensor conv2d(GpuTensor input, GpuTensor kernel, int stride, int padding) {
        int[] is = input.shape();
        int[] ks = kernel.shape();
        if (is.length != 4 || ks.length != 4) {
            throw new IllegalArgumentException("conv2d requires 4D input [N,C,H,W] and kernel [O,C,KH,KW]");
        }
        int n = is[0];
        int inC = is[1];
        int h = is[2];
        int w = is[3];
        int outC = ks[0];
        int kH = ks[2];
        int kW = ks[3];
        if (ks[1] != inC) {
            throw new IllegalArgumentException("kernel in_channels mismatch: " + ks[1] + " vs " + inC);
        }
        int outH = (h + 2 * padding - kH) / stride + 1;
        int outW = (w + 2 * padding - kW) / stride + 1;
        float[] hi = input.toHost();
        float[] hk = kernel.toHost();
        float[] hc = new float[n * outC * outH * outW];
        for (int ni = 0; ni < n; ni++) {
            for (int oi = 0; oi < outC; oi++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float sum = 0f;
                        for (int ci = 0; ci < inC; ci++) {
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                        int inIdx = ((ni * inC + ci) * h + ih) * w + iw;
                                        int kIdx = ((oi * inC + ci) * kH + kh) * kW + kw;
                                        sum += hi[inIdx] * hk[kIdx];
                                    }
                                }
                            }
                        }
                        int outIdx = ((ni * outC + oi) * outH + oh) * outW + ow;
                        hc[outIdx] = sum;
                    }
                }
            }
        }
        GpuTensor r = allocate(new int[]{n, outC, outH, outW});
        r.buffer().copyFromHost(hc, 0, 0, hc.length);
        return r;
    }

    public GpuTensor maxPool2d(GpuTensor input, int kernelSize, int stride, int padding) {
        int[] is = input.shape();
        if (is.length != 4) {
            throw new IllegalArgumentException("maxPool2d requires 4D input [N,C,H,W]");
        }
        int n = is[0];
        int c = is[1];
        int h = is[2];
        int w = is[3];
        int outH = (h + 2 * padding - kernelSize) / stride + 1;
        int outW = (w + 2 * padding - kernelSize) / stride + 1;
        float[] hi = input.toHost();
        float[] hc = new float[n * c * outH * outW];
        for (int ni = 0; ni < n; ni++) {
            for (int ci = 0; ci < c; ci++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float maxVal = Float.NEGATIVE_INFINITY;
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                    int inIdx = ((ni * c + ci) * h + ih) * w + iw;
                                    if (hi[inIdx] > maxVal) maxVal = hi[inIdx];
                                }
                            }
                        }
                        int outIdx = ((ni * c + ci) * outH + oh) * outW + ow;
                        hc[outIdx] = maxVal;
                    }
                }
            }
        }
        GpuTensor r = allocate(new int[]{n, c, outH, outW});
        r.buffer().copyFromHost(hc, 0, 0, hc.length);
        return r;
    }

    public GpuTensor mseLoss(GpuTensor pred, GpuTensor target) {
        assertSameShape(pred, target);
        int n = pred.numel();
        float[] hp = pred.toHost();
        float[] ht = target.toHost();
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            float diff = hp[i] - ht[i];
            sum += diff * diff;
        }
        GpuTensor r = allocate(new int[]{1});
        r.buffer().copyFromHost(new float[]{sum}, 0, 0, 1);
        return r;
    }

    public GpuDevice device() {
        return device;
    }

    private void cpuBinary(GpuTensor dst, GpuTensor a, GpuTensor b, FloatBinary op) {
        float[] ha = a.toHost();
        float[] hb = b.toHost();
        float[] hc = new float[ha.length];
        for (int i = 0; i < ha.length; i++) {
            hc[i] = op.apply(ha[i], hb[i]);
        }
        dst.buffer().copyFromHost(hc, 0, 0, hc.length);
    }

    private void cpuUnary(GpuTensor dst, GpuTensor a, FloatUnary op) {
        float[] ha = a.toHost();
        float[] hc = new float[ha.length];
        for (int i = 0; i < ha.length; i++) {
            hc[i] = op.apply(ha[i]);
        }
        dst.buffer().copyFromHost(hc, 0, 0, hc.length);
    }

    @FunctionalInterface
    private interface FloatBinary {
        float apply(float a, float b);
    }

    @FunctionalInterface
    private interface FloatUnary {
        float apply(float x);
    }

    private String kernelSourceFor(String op) {
        if (device instanceof com.github.tensor4j.gpu.cuda.CudaDevice ||
            device instanceof com.github.tensor4j.gpu.hip.HipDevice) {
            return cudaSource(op);
        }
        if (device instanceof com.github.tensor4j.gpu.opencl.OclDevice) {
            return openclSource(op);
        }
        return cudaSource(op);
    }

    private static String cudaSource(String op) {
        switch (op) {
            case "add": return KernelSource.addCuda();
            case "sub": return KernelSource.subCuda();
            case "mul": return KernelSource.mulCuda();
            case "div": return KernelSource.divCuda();
            case "relu": return KernelSource.reluCuda();
            case "neg": return KernelSource.negCuda();
            case "scale": return KernelSource.scaleCuda();
            case "exp": return KernelSource.expCuda();
            case "log": return KernelSource.logCuda();
            case "matmul": return KernelSource.matmulCuda();
            case "sqrt": return KernelSource.sqrtCuda();
            case "sigmoid": return KernelSource.sigmoidCuda();
            case "tanh": return KernelSource.tanhCuda();
            case "squaredDifference": return KernelSource.squaredDifferenceCuda();
            case "sum": return KernelSource.sumCuda();
            case "max": return KernelSource.maxCuda();
            case "pow": return KernelSource.powCuda();
            case "mean": return KernelSource.meanCuda();
            case "abs": return KernelSource.absCuda();
            case "clip": return KernelSource.clipCuda();
            case "transpose": return KernelSource.transposeCuda();
            case "sin": return KernelSource.sinCuda();
            case "cos": return KernelSource.cosCuda();
            case "gelu": return KernelSource.geluCuda();
            case "leakyRelu": return KernelSource.leakyReluCuda();
            case "expand": return KernelSource.expandCuda();
            case "softmax2d": return KernelSource.softmax2dCuda();
            case "silu": return KernelSource.siluCuda();
            case "elu": return KernelSource.eluCuda();
            case "subScalar": return KernelSource.subScalarCuda();
            case "binaryCrossEntropy": return KernelSource.binaryCrossEntropyCuda();
            case "dropout": return KernelSource.dropoutCuda();
            default: throw new IllegalArgumentException("unknown op: " + op);
        }
    }

    private static String openclSource(String op) {
        switch (op) {
            case "add": return KernelSource.addOpencl();
            case "sub": return KernelSource.subOpencl();
            case "mul": return KernelSource.mulOpencl();
            case "div": return KernelSource.divOpencl();
            case "relu": return KernelSource.reluOpencl();
            case "neg": return KernelSource.negOpencl();
            case "scale": return KernelSource.scaleOpencl();
            case "exp": return KernelSource.expOpencl();
            case "log": return KernelSource.logOpencl();
            case "matmul": return KernelSource.matmulOpencl();
            case "sqrt": return KernelSource.sqrtOpencl();
            case "sigmoid": return KernelSource.sigmoidOpencl();
            case "tanh": return KernelSource.tanhOpencl();
            case "squaredDifference": return KernelSource.squaredDifferenceOpencl();
            case "sum": return KernelSource.sumOpencl();
            case "max": return KernelSource.maxOpencl();
            case "pow": return KernelSource.powOpencl();
            case "mean": return KernelSource.meanOpencl();
            case "abs": return KernelSource.absOpencl();
            case "clip": return KernelSource.clipOpencl();
            case "transpose": return KernelSource.transposeOpencl();
            case "sin": return KernelSource.sinOpencl();
            case "cos": return KernelSource.cosOpencl();
            case "gelu": return KernelSource.geluOpencl();
            case "leakyRelu": return KernelSource.leakyReluOpencl();
            case "expand": return KernelSource.expandOpencl();
            case "softmax2d": return KernelSource.softmax2dOpencl();
            case "silu": return KernelSource.siluOpencl();
            case "elu": return KernelSource.eluOpencl();
            case "subScalar": return KernelSource.subScalarOpencl();
            case "binaryCrossEntropy": return KernelSource.binaryCrossEntropyOpencl();
            case "dropout": return KernelSource.dropoutOpencl();
            default: throw new IllegalArgumentException("unknown op: " + op);
        }
    }

    private void launch1D(int n, GpuProgram prog, String kernelName,
                          GpuBuffer[] bufs, int[] ints, float[] floats) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        GpuKernel kernel = prog.createKernel(kernelName);
        kernel.launch(device.createStream(), blocks, 1, 1, threads, 1, 1, bufs, ints, floats);
    }

    private void launch2D(int rows, int cols, GpuProgram prog, String kernelName,
                          GpuBuffer[] bufs, int[] ints, float[] floats) {
        int threadsX = 16;
        int threadsY = 16;
        int blocksX = (cols + threadsX - 1) / threadsX;
        int blocksY = (rows + threadsY - 1) / threadsY;
        GpuKernel kernel = prog.createKernel(kernelName);
        kernel.launch(device.createStream(), blocksX, blocksY, 1, threadsX, threadsY, 1, bufs, ints, floats);
    }

    private GpuTensor allocate(int[] shape) {
        int numel = 1;
        for (int d : shape) {
            numel *= d;
        }
        GpuBuffer buf = device.allocate((long) numel * 4L);
        int[] strides = new int[shape.length];
        int s = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = s;
            s *= shape[i];
        }
        return new GpuTensor(buf, shape, strides, 0);
    }

    private static void assertSameShape(GpuTensor a, GpuTensor b) {
        int[] sa = a.shape();
        int[] sb = b.shape();
        if (sa.length != sb.length) {
            throw new IllegalArgumentException("shape mismatch");
        }
        for (int i = 0; i < sa.length; i++) {
            if (sa[i] != sb[i]) {
                throw new IllegalArgumentException("shape mismatch at dim " + i);
            }
        }
    }

    private static GpuBuffer[] bufs(GpuTensor... t) {
        GpuBuffer[] b = new GpuBuffer[t.length];
        for (int i = 0; i < t.length; i++) {
            b[i] = t[i].buffer();
        }
        return b;
    }
    private static int[] ints(int... v) { return v; }
    private static float[] floats(float... v) { return v; }
}
