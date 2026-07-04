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

import com.github.tensor4j.core.Pool2dMath;
import com.github.tensor4j.core.Tensor;

public final class Pool2dFunction extends Function {

    private final int[] arg;
    private int[] argmax;

    public Pool2dFunction(Tensor input, int[] arg) {
        super(input);
        this.arg = arg.clone();
    }

    @Override
    public Tensor forward() {
        Pool2dMath.ForwardResult result = Pool2dMath.forwardWithMeta(inputs[0], arg);
        argmax = result.argmax;
        return result.output;
    }

    @Override
    public void backward(Tensor gradOutput) {
        if (inputs[0].requiresGrad()) {
            accumulate(inputs[0], Pool2dMath.gradInput(gradOutput, arg, inputs[0].shape().dims(), argmax));
        }
    }
}
