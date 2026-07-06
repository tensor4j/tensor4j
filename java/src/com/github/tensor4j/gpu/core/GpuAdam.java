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

import java.util.ArrayList;
import java.util.List;

/**
 * AdamW optimizer for GPU parameters.
 * Maintains first/second moment estimates on device.
 */
public final class GpuAdam {

    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final GpuDevice device;
    private final GpuTensorMath math;
    private final List<GpuTensor> mBuffer;
    private final List<GpuTensor> vBuffer;
    private int stepCount;

    public GpuAdam(GpuDevice device, float learningRate) {
        this(device, learningRate, 0.9f, 0.999f, 1e-8f, 0f);
    }

    public GpuAdam(GpuDevice device, float learningRate, float beta1, float beta2,
                   float epsilon, float weightDecay) {
        this.device = device;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.math = new GpuTensorMath(device);
        this.mBuffer = new ArrayList<>();
        this.vBuffer = new ArrayList<>();
    }

    public void step(List<GpuAutogradTensor> params) {
        stepCount++;
        float biasCorr1 = 1f - (float) Math.pow(beta1, stepCount);
        float biasCorr2 = 1f - (float) Math.pow(beta2, stepCount);
        float correctedLr = learningRate * (float) Math.sqrt(biasCorr2) / biasCorr1;

        for (int i = 0; i < params.size(); i++) {
            GpuAutogradTensor p = params.get(i);
            if (p.grad() == null) continue;

            ensureState(i, p);

            GpuTensor grad = p.grad();
            GpuTensor m = mBuffer.get(i);
            GpuTensor v = vBuffer.get(i);

            GpuTensor mNew = math.add(math.scale(m, beta1), math.scale(grad, 1f - beta1));
            GpuTensor gSq = math.mul(grad, grad);
            GpuTensor vNew = math.add(math.scale(v, beta2), math.scale(gSq, 1f - beta2));

            mBuffer.set(i, mNew);
            vBuffer.set(i, vNew);
            m.close();
            v.close();

            GpuTensor sqrtV = math.sqrt(vNew);
            int numel = sqrtV.numel();
            float[] epsArr = new float[numel];
            java.util.Arrays.fill(epsArr, epsilon);
            GpuTensor epsT = GpuTensor.fromHost(device, epsArr, sqrtV.shape());
            GpuTensor denom = math.add(sqrtV, epsT);
            sqrtV.close();
            epsT.close();
            GpuTensor step = math.div(mNew, denom);
            denom.close();
            GpuTensor biasStep = math.scale(step, correctedLr);

            if (weightDecay > 0f) {
                GpuTensor decayed = math.scale(p.data(), 1f - learningRate * weightDecay);
                GpuTensor upd = math.sub(decayed, biasStep);
                p.setData(upd);
            } else {
                GpuTensor upd = math.sub(p.data(), biasStep);
                p.setData(upd);
            }
        }
    }

    public void zeroGrad(List<GpuAutogradTensor> params) {
        for (GpuAutogradTensor p : params) {
            p.zeroGrad();
        }
    }

    private void ensureState(int index, GpuAutogradTensor p) {
        while (mBuffer.size() <= index) {
            int numel = p.data().numel();
            GpuBuffer mb = device.allocate((long) numel * 4L);
            GpuBuffer vb = device.allocate((long) numel * 4L);
            float[] zeros = new float[numel];
            mb.copyFromHost(zeros, 0, 0, numel);
            vb.copyFromHost(zeros, 0, 0, numel);
            int[] shape = p.data().shape();
            int[] strides = computeStrides(shape);
            mBuffer.add(new GpuTensor(mb, shape, strides, 0));
            vBuffer.add(new GpuTensor(vb, shape, strides, 0));
        }
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int s = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = s;
            s *= shape[i];
        }
        return strides;
    }

    public float learningRate() {
        return learningRate;
    }
}
