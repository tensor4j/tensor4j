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

import com.github.tensor4j.core.GroupNormMath;
import com.github.tensor4j.core.Tensor;

public final class GroupNormFunction extends Function {

    private final int numGroups;
    private final int channelAxis;
    private final float eps;
    private GroupNormMath.Cache cache;

    public GroupNormFunction(Tensor input, Tensor weight, Tensor bias, int numGroups, int channelAxis, float eps) {
        super(input, weight, bias);
        this.numGroups = numGroups;
        this.channelAxis = channelAxis;
        this.eps = eps;
    }

    @Override
    public Tensor forward() {
        GroupNormMath.ForwardResult result = GroupNormMath.forward(inputs[0], inputs[1], inputs[2], numGroups,
                channelAxis, eps);
        cache = result.cache;
        return result.output;
    }

    @Override
    public void backward(Tensor gradOutput) {
        int[] shape = inputs[0].shape().dims();
        if (inputs[0].requiresGrad()) {
            accumulate(inputs[0],
                    GroupNormMath.gradInput(gradOutput, inputs[1], numGroups, channelAxis, eps, cache, shape));
        }
        if (inputs[1].requiresGrad()) {
            accumulate(inputs[1], GroupNormMath.gradWeight(gradOutput, cache, shape, channelAxis));
        }
        if (inputs[2].requiresGrad()) {
            accumulate(inputs[2], GroupNormMath.gradBias(gradOutput, shape, channelAxis));
        }
    }
}
