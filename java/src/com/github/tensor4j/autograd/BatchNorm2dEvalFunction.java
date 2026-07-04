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

public final class BatchNorm2dEvalFunction extends Function {

    private final float eps;

    public BatchNorm2dEvalFunction(Tensor input, Tensor weight, Tensor bias, Tensor mean, Tensor var, float eps) {
        super(input, weight, bias, mean, var);
        this.eps = eps;
    }

    @Override
    public Tensor forward() {
        return BatchNorm2dMath.forwardEval(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps);
    }

    @Override
    public void backward(Tensor gradOutput) {
        int[] shape = inputs[0].shape().dims();
        if (inputs[0].requiresGrad()) {
            accumulate(inputs[0], BatchNorm2dMath.gradInputEval(gradOutput, inputs[1], inputs[4], eps, shape));
        }
        if (inputs[1].requiresGrad()) {
            float[] gradW = new float[inputs[1].numel()];
            float[] gradOut = gradOutput.toFlatArray();
            float[] in = inputs[0].toFlatArray();
            float[] mu = inputs[3].toFlatArray();
            float[] sigma2 = inputs[4].toFlatArray();
            for (int index = 0; index < gradOut.length; index++) {
                int c = (index / (shape[2] * shape[3])) % shape[1];
                float xhat = (in[index] - mu[c]) / (float) Math.sqrt(sigma2[c] + eps);
                gradW[c] += gradOut[index] * xhat;
            }
            accumulate(inputs[1], Tensor.of(gradW, inputs[1].shape().dims()));
        }
        if (inputs[2].requiresGrad()) {
            accumulate(inputs[2], BatchNorm2dMath.gradBias(gradOutput, shape));
        }
    }
}
