/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.nn;

import com.github.tensor4j.autograd.BiasAddFunction;
import com.github.tensor4j.core.Tensor;

/** Affine helpers without lambdas. */
final class LinearOps {

    private LinearOps() {
    }

    static Tensor addBias(Tensor activations, Tensor bias) {
        int batchSize = activations.shape().dim(0);
        int outFeatures = activations.shape().dim(1);
        float[] output = new float[activations.numel()];
        float[] activationData = activations.data();
        float[] biasData = bias.data();
        for (int batch = 0; batch < batchSize; batch++) {
            for (int feature = 0; feature < outFeatures; feature++) {
                int index = batch * outFeatures + feature;
                output[index] = activationData[index] + biasData[feature];
            }
        }
        Tensor result = Tensor.of(output, batchSize, outFeatures);
        if (activations.requiresGrad() || bias.requiresGrad()) {
            result.withGrad(true);
            result.setCreator(new BiasAddFunction(activations, bias, batchSize, outFeatures), activations, bias);
        }
        return result;
    }
}
