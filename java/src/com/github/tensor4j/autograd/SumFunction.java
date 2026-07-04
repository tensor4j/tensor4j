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

import java.util.Arrays;
import com.github.tensor4j.core.Tensor;

/** Gradient for sum reduction ({@code dX = broadcast(dY)}). */
public final class SumFunction extends Function {

    public SumFunction(Tensor input) {
        super(input);
    }

    @Override
    public Tensor forward() {
        return inputs[0].sum();
    }

    @Override
    public void backward(Tensor gradOutput) {
        int n = inputs[0].numel();
        float scale = gradOutput.getFlat(0);
        float[] grad = new float[n];
        Arrays.fill(grad, scale);
        accumulate(inputs[0], Tensor.of(grad, inputs[0].shape().dims()));
    }
}
