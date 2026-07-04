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

import com.github.tensor4j.core.SoftmaxMath;
import com.github.tensor4j.core.Tensor;

public final class LogSoftmaxFunction extends Function {

    private final int axis;
    private Tensor savedOutput;

    public LogSoftmaxFunction(Tensor input, int axis) {
        super(input);
        this.axis = axis;
    }

    @Override
    public Tensor forward() {
        savedOutput = SoftmaxMath.logSoftmax(inputs[0], axis);
        return savedOutput;
    }

    @Override
    public void backward(Tensor gradOutput) {
        accumulate(inputs[0], SoftmaxMath.logSoftmaxGradInput(gradOutput, savedOutput, axis));
    }
}
