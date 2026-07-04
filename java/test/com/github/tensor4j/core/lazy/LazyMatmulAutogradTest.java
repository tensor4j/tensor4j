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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Layer 2n: lazy matmul autograd via UOp gradient rules (decomposed dot backward). */
class LazyMatmulAutogradTest {

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void matmulSumBackwardMatchesEager() {
        float[] aData = new float[] {1f, 2f, 3f, 4f};
        float[] bData = new float[] {5f, 6f, 7f, 8f};
        Tensor a = Tensor.of(aData, 2, 2).withGrad(true);
        Tensor b = Tensor.of(bData, 2, 2).withGrad(true);
        a.matmul(b).sum().backward();

        LazyTensor la = LazyTensor.of(aData, 2, 2).withGrad(true);
        LazyTensor lb = LazyTensor.of(bData, 2, 2).withGrad(true);
        la.matmul(lb).sum().backward();

        assertAllClose(a.grad(), la.grad(), 1e-5f);
        assertAllClose(b.grad(), lb.grad(), 1e-5f);
    }

    @Test
    void matmulChainWithReluMatchesEagerGrad() {
        float[] aData = new float[] {1f, -1f, 2f, 3f};
        float[] bData = new float[] {1f, 0f, 0f, 1f};
        Tensor a = Tensor.of(aData, 2, 2).withGrad(true);
        Tensor b = Tensor.of(bData, 2, 2).withGrad(true);
        a.matmul(b).relu().sum().backward();

        LazyTensor la = LazyTensor.of(aData, 2, 2).withGrad(true);
        LazyTensor lb = LazyTensor.of(bData, 2, 2).withGrad(true);
        la.matmul(lb).relu().sum().backward();

        assertGradPresent(la.leafTensor());
        assertAllClose(a.grad(), la.grad(), 1e-5f);
        assertAllClose(b.grad(), lb.grad(), 1e-5f);
    }
}
