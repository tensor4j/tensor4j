/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.infer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import org.junit.jupiter.api.Test;

class RopeTest {

    @Test
    void rotationChangesVector() {
        InferTensor input = InferTensor.vector(1, 0, 0, 0);
        InferTensor out = Rope.applyHeads(input, 1, 4, new int[] {1},
                RopeConfig.llamaDefaults(4));
        assertNotEquals(input.get(0, 0), out.get(0, 0), 1e-6f);
    }

    @Test
    void positionZeroIsIdentity() {
        InferTensor input = InferTensor.vector(1, 2, 0, 0);
        InferTensor out = Rope.applyHeads(input, 1, 4, new int[] {0},
                RopeConfig.llamaDefaults(4));
        assertEquals(input.get(0, 0), out.get(0, 0), 1e-5f);
        assertEquals(input.get(0, 1), out.get(0, 1), 1e-5f);
    }

    @Test
    void partialRopeLeavesTailUntouched() {
        RopeConfig partial = new RopeConfig(10000.0f, 1.0f, 2, RopeScalingType.NONE, 0.0f, 1.0f, 32.0f, 1.0f, 0);
        InferTensor input = InferTensor.vector(1, 2, 3, 4);
        InferTensor out = Rope.applyHeads(input, 1, 4, new int[] {3}, partial);
        assertEquals(3.0f, out.get(0, 2), 1e-5f);
        assertEquals(4.0f, out.get(0, 3), 1e-5f);
    }

    @Test
    void yarnDiffersFromStandardAtHighPosition() {
        RopeConfig standard = new RopeConfig(10000.0f, 1.0f, 4, RopeScalingType.NONE, 0.0f, 1.0f, 32.0f, 1.0f, 0);
        RopeConfig yarn = new RopeConfig(10000.0f, 0.5f, 4, RopeScalingType.YARN, 1.0f, 1.0f, 32.0f, 1.0f, 4);
        InferTensor input = InferTensor.vector(1, 0, 0, 0);
        InferTensor stdOut = Rope.applyHeads(input, 1, 4, new int[] {32}, standard);
        InferTensor yarnOut = Rope.applyHeads(input, 1, 4, new int[] {32}, yarn);
        assertNotEquals(stdOut.get(0, 0), yarnOut.get(0, 0), 1e-4f);
    }

    @Test
    void disabledRopeIsNoOp() {
        InferTensor input = InferTensor.vector(3, 4, 0, 0);
        InferTensor out = Rope.applyHeads(input, 1, 4, new int[] {5}, RopeConfig.disabled());
        assertEquals(input.get(0, 0), out.get(0, 0), 0.0f);
    }
}
