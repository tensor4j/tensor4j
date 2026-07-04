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

import java.util.List;
import com.github.tensor4j.core.Tensor;

/** Affine layer {@code y = xW + b} (tinygrad {@code nn.Linear}). */
public final class Linear extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final String name;

    public Linear(int inFeatures, int outFeatures, String name) {
        this.name = name;
        float scale = (float) Math.sqrt(2.0 / inFeatures);
        weight = Tensor.randn(inFeatures, outFeatures).withGrad(true);
        bias = Tensor.zeros(outFeatures).withGrad(true);
        for (int i = 0; i < weight.mutableData().length; i++) {
            weight.mutableData()[i] *= scale;
        }
    }

    public Linear(Tensor weight, Tensor bias, String name) {
        this.weight = weight.withGrad(true);
        this.bias = bias.withGrad(true);
        this.name = name;
    }

    public String name() {
        return name;
    }

    public Tensor weight() {
        return weight;
    }

    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor batch = input.shape().rank() == 1 ? input.reshape(1, input.shape().dim(0)) : input;
        return LinearOps.addBias(batch.matmul(weight), bias);
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weight, bias);
    }
}
