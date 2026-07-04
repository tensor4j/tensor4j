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

/** Elementwise division autograd ({@code d(a/b) = da/b, db = -da/b^2}). */
public final class DivFunction extends Function {

    public DivFunction(Tensor left, Tensor right) {
        super(left, right);
    }

    @Override
    public Tensor forward() {
        return inputs[0].div(inputs[1]);
    }

    @Override
    public void backward(Tensor gradOutput) {
        Tensor left = inputs[0];
        Tensor right = inputs[1];
        float[] grad = gradOutput.contiguous().toFlatArray();
        float[] leftValues = left.contiguous().toFlatArray();
        float[] rightValues = right.contiguous().toFlatArray();
        float[] gradLeft = new float[grad.length];
        float[] gradRight = new float[grad.length];
        for (int i = 0; i < grad.length; i++) {
            float denominator = rightValues[i];
            gradLeft[i] = grad[i] / denominator;
            gradRight[i] = -grad[i] * leftValues[i] / (denominator * denominator);
        }
        accumulate(inputs[0], Tensor.of(gradLeft, left.shape().dims()));
        accumulate(inputs[1], Tensor.of(gradRight, right.shape().dims()));
    }
}
