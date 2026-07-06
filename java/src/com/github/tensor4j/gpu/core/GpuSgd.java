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

import java.util.List;

/**
 * SGD optimizer for GPU parameters (w = w - lr * grad).
 */
public final class GpuSgd {

    private final float learningRate;
    private final float weightDecay;
    private final float momentum;
    private final GpuDevice device;
    private final GpuTensorMath math;

    public GpuSgd(GpuDevice device, float learningRate) {
        this(device, learningRate, 0f, 0f);
    }

    public GpuSgd(GpuDevice device, float learningRate, float weightDecay, float momentum) {
        this.device = device;
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.momentum = momentum;
        this.math = new GpuTensorMath(device);
    }

    public void step(List<GpuAutogradTensor> params) {
        for (GpuAutogradTensor p : params) {
            if (p.grad() == null) continue;
            GpuTensor grad = p.grad();
            GpuTensor update;
            if (weightDecay > 0f) {
                GpuTensor decayed = math.scale(p.data(), -learningRate * weightDecay);
                GpuTensor gradStep = math.scale(grad, -learningRate);
                update = math.add(decayed, gradStep);
            } else {
                update = math.scale(grad, -learningRate);
            }
            GpuTensor newData = math.add(p.data(), update);
            p.setData(newData);
        }
    }

    public void zeroGrad(List<GpuAutogradTensor> params) {
        for (GpuAutogradTensor p : params) {
            p.zeroGrad();
        }
    }
}
