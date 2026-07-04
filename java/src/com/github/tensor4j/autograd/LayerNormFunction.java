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

import com.github.tensor4j.core.LayerNormMath;
import com.github.tensor4j.core.Tensor;

public final class LayerNormFunction extends Function {

    private final int[] normalizedShape;
    private final float eps;
    private LayerNormMath.Cache cache;

    public LayerNormFunction(Tensor input, Tensor weight, Tensor bias, int[] normalizedShape, float eps) {
        super(input, weight, bias);
        this.normalizedShape = normalizedShape.clone();
        this.eps = eps;
    }

    @Override
    public Tensor forward() {
        LayerNormMath.ForwardResult result = LayerNormMath.forward(inputs[0], inputs[1], inputs[2], normalizedShape,
                eps);
        cache = result.cache;
        return result.output;
    }

    @Override
    public void backward(Tensor gradOutput) {
        int[] shape = inputs[0].shape().dims();
        if (inputs[0].requiresGrad()) {
            accumulate(inputs[0], LayerNormMath.gradInput(gradOutput, inputs[1], normalizedShape, eps, cache, shape));
        }
        if (inputs[1].requiresGrad()) {
            accumulate(inputs[1], LayerNormMath.gradWeight(gradOutput, cache, normalizedShape));
        }
        if (inputs[2].requiresGrad()) {
            accumulate(inputs[2], LayerNormMath.gradBias(gradOutput, normalizedShape));
        }
    }
}
