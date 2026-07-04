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

/** Gradient for reshape view ({@code dX = reshape(dY, X.shape)}). */
public final class ReshapeFunction extends Function {

    private final int[] inputShape;

    public ReshapeFunction(Tensor input, int[] inputShape) {
        super(input);
        this.inputShape = inputShape.clone();
    }

    @Override
    public Tensor forward() {
        throw new UnsupportedOperationException("forward handled by Tensor.reshape");
    }

    @Override
    public void backward(Tensor gradOutput) {
        accumulate(inputs[0], gradOutput.reshape(inputShape));
    }
}
