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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Layer 1k: {@link LazyUOp} DAG structure (tinygrad {@code UOp} graph semantics). */
class LazyGraphTest {

    @BeforeEach
    void clearInternCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void binaryOpHasTwoParentsInSrc() {
        LazyTensor left = LazyTensor.of(new float[] {1f, 2f}, 2);
        LazyTensor right = LazyTensor.of(new float[] {3f, 4f}, 2);
        LazyUOp add = left.add(right).uop();
        assertEquals(LazyUOp.Kind.ADD, add.op());
        assertEquals(2, add.srcCount());
        assertSame(left.uop(), add.src(0));
        assertSame(right.uop(), add.src(1));
    }

    @Test
    void reusedSubgraphIsInternedToSameNode() {
        LazyTensor x = LazyTensor.of(new float[] {2f, 3f}, 2);
        LazyTensor square = x.mul(x);
        LazyTensor loss = square.add(square);
        assertSame(square.uop(), loss.uop().src(0));
        assertSame(square.uop(), loss.uop().src(1));
        assertEquals(3, loss.graphNodeCount());
    }

    @Test
    void toposortVisitsSharedNodeOnce() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f}, 2);
        LazyTensor square = x.mul(x);
        LazyTensor loss = square.add(square);
        assertEquals(3, LazyGraph.toposort(loss.uop()).size());
    }

    @Test
    void diamondGraphDepthIsTwo() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f, 3f}, 3);
        LazyTensor y = x.relu();
        LazyTensor z = x.neg();
        LazyTensor loss = y.add(z);
        assertEquals(2, loss.graphDepth());
        assertEquals(4, loss.graphNodeCount());
        assertTrue(LazyGraph.references(loss.uop(), x.uop()));
    }

    @Test
    void realizeComputesSharedSubgraphOnce() {
        LazyTensor x = LazyTensor.of(new float[] {2f, 3f}, 2).withGrad(true);
        LazyTensor square = x.mul(x);
        LazyTensor loss = square.add(square).sum();
        loss.backward();
        assertAllClose(new float[] {8f, 12f}, x.grad(), 1e-6f);
    }

    @Test
    void deepwalkPrunesUnusedSiblingBranch() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f}, 2).withGrad(true);
        LazyTensor y = LazyTensor.of(new float[] {3f, 4f}, 2).withGrad(true);
        LazyUOp loss = x.add(y).sum().uop();
        java.util.Set<LazyUOp> targets = java.util.Set.of(x.uop());
        java.util.List<LazyUOp> walk = LazyGraph.deepwalk(loss, targets);
        java.util.List<LazyUOp> full = LazyGraph.toposort(loss);
        assertTrue(full.contains(y.uop()));
        assertTrue(walk.contains(x.uop()));
        assertTrue(!walk.contains(y.uop()), "deepwalk should skip sibling not on path to target");
    }
}
