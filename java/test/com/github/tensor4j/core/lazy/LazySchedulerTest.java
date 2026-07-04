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

/** Layer 1l: scheduler + elementwise fusion (tinygrad schedule/fusion subset). */
class LazySchedulerTest {

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void mulReluFusesIntoOneKernel() {
        LazyTensor x = LazyTensor.of(new float[] {1f, -2f, 3f}, 3);
        LazyTensor y = LazyTensor.of(new float[] {2f, 2f, 2f}, 3);
        LazySchedule schedule = x.mul(y).relu().schedule();
        assertEquals(1, schedule.kernelCount());
        assertTrue(schedule.savedKernels() >= 1);
        assertEquals(LazyKernel.Kind.FUSED, schedule.kernels().get(0).kind());
        assertEquals(4, schedule.kernels().get(0).opCount());
    }

    @Test
    void sumBreaksFusionAfterElementwiseChain() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f}, 2);
        LazySchedule schedule = x.mul(x).relu().sum().schedule();
        assertEquals(2, schedule.kernelCount());
    }

    @Test
    void sharedSubgraphIsNotFusedIntoAdd() {
        LazyTensor x = LazyTensor.of(new float[] {2f, 3f}, 2);
        LazyTensor square = x.mul(x);
        LazySchedule schedule = square.add(square).sum().schedule();
        assertEquals(3, schedule.kernelCount());
    }

    @Test
    void movementOpsScheduleSeparately() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2);
        LazySchedule schedule = x.permute(1, 0).relu().schedule();
        assertEquals(2, schedule.kernelCount());
    }

    @Test
    void scheduledExecuteMatchesDirectRealize() {
        LazyTensor lazy = LazyTensor.of(new float[] {1f, -2f, 3f, 4f}, 4)
                .mul(LazyTensor.of(new float[] {2f, 2f, 2f, 2f}, 4))
                .relu();
        assertAllClose(lazy.realize(), lazy.schedule().execute(), 1e-6f);
    }
}
