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

import com.github.tensor4j.core.MaxUnpool2dMath;
import com.github.tensor4j.core.Tensor;

public final class MaxUnpool2dFunction extends Function {

    private final int[] poolArg;
    private final int[] outputShape;

    public MaxUnpool2dFunction(Tensor values, Tensor indices, int[] poolArg, int[] outputShape) {
        super(values, indices);
        this.poolArg = poolArg.clone();
        this.outputShape = outputShape.clone();
    }

    @Override
    public Tensor forward() {
        return MaxUnpool2dMath.forward(inputs[0], inputs[1], poolArg, outputShape);
    }

    @Override
    public void backward(Tensor gradOutput) {
        if (inputs[0].requiresGrad()) {
            accumulate(inputs[0], MaxUnpool2dMath.gradValues(gradOutput, inputs[1], poolArg, inputs[0].shape().dims()));
        }
    }
}
