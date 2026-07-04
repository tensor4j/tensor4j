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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.Test;

/**
 * Layer 2l: lazy UOp autograd (tinygrad {@code compute_gradient} on {@link LazyUOp} DAG).
 */
class LazyAutogradTest {

    @Test
    void backwardRealizesGraphAndPopulatesLeafGrad() {
        LazyTensor x = LazyTensor.of(new float[] {2f, 3f}, 2).withGrad(true);
        LazyTensor loss = x.mul(x).sum();
        assertFalse(loss.isRealized());
        loss.backward();
        assertTrue(loss.isRealized());
        assertGradPresent(x.leafTensor());
        assertAllClose(new float[] {4f, 6f}, x.grad());
    }

    @Test
    void addTwoLeavesAccumulatesUnitGradients() {
        LazyTensor left = LazyTensor.of(new float[] {1f, 2f}, 2).withGrad(true);
        LazyTensor right = LazyTensor.of(new float[] {3f, 4f}, 2).withGrad(true);
        LazyTensor loss = left.add(right).sum();
        loss.backward();
        assertAllClose(new float[] {1f, 1f}, left.grad());
        assertAllClose(new float[] {1f, 1f}, right.grad());
    }

    @Test
    void reluBlocksNegativeInputGradientThroughLazyChain() {
        LazyTensor x = LazyTensor.of(new float[] {-1f, 2f}, 2).withGrad(true);
        LazyTensor loss = x.relu().sum();
        loss.backward();
        assertNotNull(x.grad());
        assertAllClose(new float[] {0f, 1f}, x.grad());
    }

    @Test
    void permuteSumBackwardPreservesTotalGradient() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2).withGrad(true);
        LazyTensor loss = x.permute(1, 0).sum();
        loss.backward();
        assertGradPresent(x.leafTensor());
        assertAllClose(new float[] {1f, 1f, 1f, 1f}, x.grad());
    }

    @Test
    void expandBackwardSumsOverBroadcastAxes() {
        LazyTensor x = LazyTensor.of(new float[] {2f, 3f}, 1, 2).withGrad(true);
        LazyTensor loss = x.expand(3, 1, 2).sum();
        loss.backward();
        assertAllClose(new float[] {3f, 3f}, x.grad());
    }

    @Test
    void broadcastAddBackwardDistributesGradToBothLeaves() {
        LazyTensor row = LazyTensor.of(new float[] {1f, 2f, 3f}, 3, 1).withGrad(true);
        LazyTensor col = LazyTensor.of(new float[] {10f, 20f}, 1, 2).withGrad(true);
        LazyTensor loss = row.expand(3, 2).add(col.expand(3, 2)).sum();
        loss.backward();
        assertAllClose(new float[] {2f, 2f, 2f}, row.grad());
        assertAllClose(new float[] {3f, 3f}, col.grad());
    }

    @Test
    void zeroGradClearsLeafBuffersInGraph() {
        LazyTensor x = LazyTensor.of(new float[] {1f}, 1).withGrad(true);
        LazyTensor loss = x.mul(x);
        loss.backward();
        assertGradPresent(x.leafTensor());
        x.zeroGrad();
        assertTrue(x.grad() == null);
    }
}
