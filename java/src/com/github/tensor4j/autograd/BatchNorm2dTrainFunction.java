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

import com.github.tensor4j.core.BatchNorm2dMath;
import com.github.tensor4j.core.Tensor;

public final class BatchNorm2dTrainFunction extends Function {

    private final float eps;
    private BatchNorm2dMath.TrainCache cache;

    public BatchNorm2dTrainFunction(Tensor input, Tensor weight, Tensor bias, float eps) {
        super(input, weight, bias);
        this.eps = eps;
    }

    @Override
    public Tensor forward() {
        BatchNorm2dMath.TrainForward result = BatchNorm2dMath.forwardTrain(inputs[0], inputs[1], inputs[2], eps);
        cache = result.cache;
        return result.output;
    }

    @Override
    public void backward(Tensor gradOutput) {
        int[] shape = inputs[0].shape().dims();
        if (inputs[0].requiresGrad()) {
            accumulate(inputs[0], BatchNorm2dMath.gradInputTrain(gradOutput, inputs[1], eps, cache, shape));
        }
        if (inputs[1].requiresGrad()) {
            accumulate(inputs[1], BatchNorm2dMath.gradWeight(gradOutput, cache, shape));
        }
        if (inputs[2].requiresGrad()) {
            accumulate(inputs[2], BatchNorm2dMath.gradBias(gradOutput, shape));
        }
    }
}
