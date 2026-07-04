/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.autograd;

import com.github.tensor4j.core.Tensor;

/** Broadcast-add bias vector to a batch matrix with autograd. */
public final class BiasAddFunction extends Function {

    private final int batchSize;
    private final int outFeatures;

    public BiasAddFunction(Tensor activations, Tensor bias, int batchSize, int outFeatures) {
        super(activations, bias);
        this.batchSize = batchSize;
        this.outFeatures = outFeatures;
    }

    @Override
    public Tensor forward() {
        throw new UnsupportedOperationException("forward is handled by Linear.forward");
    }

    @Override
    public void backward(Tensor gradOutput) {
        accumulate(inputs[0], gradOutput);
        float[] biasGrad = new float[outFeatures];
        for (int batch = 0; batch < batchSize; batch++) {
            for (int feature = 0; feature < outFeatures; feature++) {
                biasGrad[feature] += gradOutput.getFlat(batch * outFeatures + feature);
            }
        }
        accumulate(inputs[1], Tensor.of(biasGrad, outFeatures));
    }
}
