/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.optim;

import java.util.List;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.core.TensorMath;

/** Stochastic gradient descent with optional global grad-norm clipping. */
public final class Sgd {

    private final float learningRate;
    private final float maxGradNorm;

    public Sgd(float learningRate) {
        this(learningRate, 1f);
    }

    public Sgd(float learningRate, float maxGradNorm) {
        this.learningRate = learningRate;
        this.maxGradNorm = maxGradNorm;
    }

    public void step(List<Tensor> parameters) {
        if (maxGradNorm > 0f) {
            clipGradNorm(parameters, maxGradNorm);
        }
        for (Tensor param : parameters) {
            param.applyGrad(learningRate);
        }
    }

    public float learningRate() {
        return learningRate;
    }

    private static void clipGradNorm(List<Tensor> parameters, float maxNorm) {
        float totalNorm = 0f;
        for (Tensor param : parameters) {
            if (param.grad() == null) {
                continue;
            }
            float[] grad = param.grad().contiguous().toFlatArray();
            totalNorm += TensorMath.l2Norm(grad, grad.length) * TensorMath.l2Norm(grad, grad.length);
        }
        totalNorm = (float) Math.sqrt(totalNorm);
        if (totalNorm <= maxNorm || totalNorm == 0f) {
            return;
        }
        float scale = maxNorm / totalNorm;
        for (Tensor param : parameters) {
            if (param.grad() == null) {
                continue;
            }
            float[] grad = param.grad().contiguous().toFlatArray();
            TensorMath.scale(grad, grad.length, scale);
            param.setGrad(Tensor.of(grad, param.shape().dims()));
        }
    }
}
