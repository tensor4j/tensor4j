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
import com.github.tensor4j.core.TensorMath;

/** Matmul backward via contiguous kernels (no autograd subgraph in backward). */
public final class MatMulFunction extends Function {

    public MatMulFunction(Tensor left, Tensor right) {
        super(left, right);
    }

    @Override
    public Tensor forward() {
        return inputs[0].matmul(inputs[1]);
    }

    @Override
    public void backward(Tensor gradOutput) {
        Tensor left = inputs[0];
        Tensor right = inputs[1];
        int batch = gradOutput.layout().dim(0);
        int outFeatures = gradOutput.layout().dim(1);
        int inFeatures = left.layout().dim(1);
        float[] gradFlat = gradOutput.contiguousFlatOrCopy();
        float[] rightFlat = right.contiguousFlatOrCopy();
        float[] leftFlat = left.contiguousFlatOrCopy();
        float[] gradLeft = TensorMath.matmulGradLeft(
                gradFlat, batch, inFeatures, outFeatures, rightFlat, right.layout().dim(0), right.layout().dim(1));
        float[] gradRight = TensorMath.matmulGradRight(
                leftFlat, left.layout().dim(0), left.layout().dim(1), gradFlat, batch, outFeatures);
        accumulate(inputs[0], Tensor.of(gradLeft, left.layout().shape()));
        accumulate(inputs[1], Tensor.of(gradRight, right.layout().shape()));
    }
}
