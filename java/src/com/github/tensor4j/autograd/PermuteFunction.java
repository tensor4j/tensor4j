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

import com.github.tensor4j.core.Strides;
import com.github.tensor4j.core.Tensor;

/** Gradient for permute view ({@code dX = permute(dY, inverse(order))}). */
public final class PermuteFunction extends Function {

    private final int[] order;

    public PermuteFunction(Tensor input, int[] order) {
        super(input);
        this.order = order.clone();
    }

    @Override
    public Tensor forward() {
        throw new UnsupportedOperationException("forward handled by Tensor.permute");
    }

    @Override
    public void backward(Tensor gradOutput) {
        accumulate(inputs[0], gradOutput.permute(Strides.inversePermute(order)));
    }
}
