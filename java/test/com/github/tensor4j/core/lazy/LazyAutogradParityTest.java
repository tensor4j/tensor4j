/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core.lazy;

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static com.github.tensor4j.support.TensorAssert.assertGradPresent;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.Test;

/**
 * Layer 2m: lazy autograd gradients match the eager {@link Tensor} path (tinygrad parity at teaching scale).
 */
class LazyAutogradParityTest {

    @Test
    void mulSquareSumMatchesEagerGrad() {
        float[] data = new float[] {2f, 3f, 4f};
        Tensor eager = Tensor.of(data, 3).withGrad(true);
        eager.mul(eager).sum().backward();

        LazyTensor lazy = LazyTensor.of(data, 3).withGrad(true);
        lazy.mul(lazy).sum().backward();

        assertAllClose(eager.grad(), lazy.grad(), 1e-6f);
    }

    @Test
    void deepMovementComputeChainMatchesEagerGrad() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor eager = Tensor.of(data, 2, 3).withGrad(true);
        eager.permute(1, 0).reshape(6).relu().sum().backward();

        LazyTensor lazy = LazyTensor.of(data, 2, 3).withGrad(true);
        lazy.permute(1, 0).reshape(6).relu().sum().backward();

        assertGradPresent(lazy.leafTensor());
        assertAllClose(eager.grad(), lazy.grad(), 1e-6f);
    }

    @Test
    void broadcastMulMatchesEagerGrad() {
        float[] leftData = new float[] {1f, 2f, 3f};
        float[] rightData = new float[] {4f, 5f, 6f};
        Tensor left = Tensor.of(leftData, 3, 1).withGrad(true);
        Tensor right = Tensor.of(rightData, 1, 3).withGrad(true);
        left.expand(3, 3).mul(right.expand(3, 3)).sum().backward();

        LazyTensor lazyLeft = LazyTensor.of(leftData, 3, 1).withGrad(true);
        LazyTensor lazyRight = LazyTensor.of(rightData, 1, 3).withGrad(true);
        lazyLeft.expand(3, 3).mul(lazyRight.expand(3, 3)).sum().backward();

        assertAllClose(left.grad(), lazyLeft.grad(), 1e-6f);
        assertAllClose(right.grad(), lazyRight.grad(), 1e-6f);
    }

    @Test
    void withGradAfterGraphBuildStillBackprops() {
        float[] data = new float[] {3f, 4f};
        LazyTensor x = LazyTensor.of(data, 2);
        LazyTensor loss = x.mul(x).sum();
        x.withGrad(true);
        loss.backward();
        assertAllClose(new float[] {6f, 8f}, x.grad(), 1e-6f);
    }

    @Test
    void lazyMatmulGradMatchesEager() {
        float[] aData = new float[] {2f, 1f, 0f, 3f};
        float[] bData = new float[] {1f, 2f, 3f, 4f};
        Tensor a = Tensor.of(aData, 2, 2).withGrad(true);
        Tensor b = Tensor.of(bData, 2, 2).withGrad(true);
        a.matmul(b).sum().backward();

        LazyTensor la = LazyTensor.of(aData, 2, 2).withGrad(true);
        LazyTensor lb = LazyTensor.of(bData, 2, 2).withGrad(true);
        la.matmul(lb).sum().backward();

        assertAllClose(a.grad(), la.grad(), 1e-5f);
        assertAllClose(b.grad(), lb.grad(), 1e-5f);
    }
}
