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

import java.util.ArrayList;
import java.util.List;
import com.github.tensor4j.core.Tensor;

/** Gradient for expand/broadcast view (sum over broadcast axes). */
public final class ExpandFunction extends Function {

    private final int[] inputShape;

    public ExpandFunction(Tensor input, int[] inputShape) {
        super(input);
        this.inputShape = inputShape.clone();
    }

    @Override
    public Tensor forward() {
        throw new UnsupportedOperationException("forward handled by Tensor.expand");
    }

    @Override
    public void backward(Tensor gradOutput) {
        Tensor grad = gradOutput.contiguous();
        int[] expandedShape = grad.shape().dims();
        int pad = expandedShape.length - inputShape.length;
        List<Integer> axes = new ArrayList<>();
        for (int axis = 0; axis < expandedShape.length; axis++) {
            int sourceDim = axis < pad ? 1 : inputShape[axis - pad];
            if (sourceDim == 1 && expandedShape[axis] > 1) {
                axes.add(axis);
            }
        }
        for (int index = axes.size() - 1; index >= 0; index--) {
            grad = grad.sumAxis(axes.get(index));
        }
        if (!java.util.Arrays.equals(grad.shape().dims(), inputShape)) {
            grad = grad.reshape(inputShape);
        }
        accumulate(inputs[0], grad);
    }
}
