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

import com.github.tensor4j.core.Conv2dMath;
import com.github.tensor4j.core.Tensor;

public final class Conv2dFunction extends Function {

    private final int[] arg;

    public Conv2dFunction(Tensor input, Tensor weight, int[] arg) {
        super(input, weight);
        this.arg = arg.clone();
    }

    @Override
    public Tensor forward() {
        return Conv2dMath.forward(inputs[0], inputs[1], arg);
    }

    @Override
    public void backward(Tensor gradOutput) {
        if (inputs[0].requiresGrad()) {
            accumulate(inputs[0], Conv2dMath.gradInput(gradOutput, inputs[1], arg, inputs[0].shape().dims()));
        }
        if (inputs[1].requiresGrad()) {
            accumulate(inputs[1], Conv2dMath.gradWeight(gradOutput, inputs[0], arg, inputs[1].shape().dims()));
        }
    }
}
