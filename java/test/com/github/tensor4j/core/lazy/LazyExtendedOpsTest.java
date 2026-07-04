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
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Layer 2q: forward realize for extended lazy ops (tinygrad pm_gradient forward subset). */
class LazyExtendedOpsTest {

    private static final float EPS = 1e-5f;

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void powSqrtLog2Exp2Forward() {
        float[] data = new float[] {2f, 4f, 9f};
        LazyTensor x = LazyTensor.of(data, 3);
        assertAllClose(new float[] {4f, 16f, 81f}, x.pow(2f).realize(), EPS);
        assertAllClose(new float[] {(float) Math.sqrt(2), 2f, 3f}, x.sqrt().realize(), EPS);
        assertAllClose(new float[] {1f, 2f, (float) (Math.log(9) / Math.log(2))}, x.log2().realize(), EPS);
        assertAllClose(new float[] {4f, 16f, 512f}, x.exp2().realize(), EPS);
    }

    @Test
    void maxWhereForward() {
        LazyTensor left = LazyTensor.of(new float[] {1f, 5f, 3f}, 3);
        LazyTensor right = LazyTensor.of(new float[] {2f, 4f, 3f}, 3);
        assertAllClose(new float[] {2f, 5f, 3f}, left.max(right).realize(), EPS);

        LazyTensor cond = LazyTensor.of(new float[] {1f, 0f, 1f}, 3);
        LazyTensor ifTrue = LazyTensor.of(new float[] {10f, 20f, 30f}, 3);
        LazyTensor ifFalse = LazyTensor.of(new float[] {1f, 2f, 3f}, 3);
        assertAllClose(new float[] {10f, 2f, 30f}, cond.where(ifTrue, ifFalse).realize(), EPS);
    }

    @Test
    void padShrinkContiguousCastForward() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f, 3f}, 3);
        LazyTensor padded = x.pad(1, 1);
        assertEquals(5, padded.shape()[0]);
        assertAllClose(new float[] {0f, 1f, 2f, 3f, 0f}, padded.realize(), EPS);

        LazyTensor shrunk = padded.shrink(1, 4);
        assertAllClose(new float[] {1f, 2f, 3f}, shrunk.realize(), EPS);

        LazyTensor copied = x.contiguous();
        assertAllClose(data(x), copied.realize(), EPS);
        assertAllClose(data(x), x.cast().realize(), EPS);
    }

    @Test
    void extendedOpsBuildExpectedUOpKinds() {
        LazyTensor x = LazyTensor.of(new float[] {2f}, 1);
        assertTrue(hasKind(x.pow(3f).uop(), LazyUOp.Kind.POW));
        assertTrue(hasKind(x.max(x).uop(), LazyUOp.Kind.MAX));
        assertTrue(hasKind(x.log2().uop(), LazyUOp.Kind.LOG2));
        assertTrue(hasKind(x.pad(0, 1).uop(), LazyUOp.Kind.PAD));
    }

    private static float[] data(LazyTensor tensor) {
        return tensor.realize().data();
    }

    private static boolean hasKind(LazyUOp root, LazyUOp.Kind kind) {
        for (LazyUOp node : LazyGraph.toposort(root)) {
            if (node.op() == kind) {
                return true;
            }
        }
        return false;
    }
}
