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

public final class ReluFunction extends Function {

    public ReluFunction(Tensor input) {
        super(input);
    }

    @Override
    public Tensor forward() {
        return inputs[0].relu();
    }

    @Override
    public void backward(Tensor gradOutput) {
        int n = inputs[0].numel();
        float[] mask = new float[n];
        for (int i = 0; i < n; i++) {
            mask[i] = inputs[0].getFlat(i) > 0f ? gradOutput.getFlat(i) : 0f;
        }
        accumulate(inputs[0], Tensor.of(mask, inputs[0].shape().dims()));
    }
}
