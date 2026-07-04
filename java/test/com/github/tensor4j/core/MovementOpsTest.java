/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core;

import static com.github.tensor4j.support.TensorAssert.assertClose;
import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static com.github.tensor4j.support.TensorAssert.assertShape;
import static com.github.tensor4j.support.TensorAssert.matrix2d;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Layer 1b: movement ops (tinygrad expand / permute / reshape / flatten). */
class MovementOpsTest {

    @Test
    void expandUsesStrideZeroBroadcast() {
        Tensor vector = Tensor.of(new float[] {1f, 2f, 3f}, 3);
        Tensor expanded = vector.expand(2, 3);
        assertShape(new int[] {2, 3}, expanded);
        assertFalse(expanded.isContiguous());
        assertAllClose(new float[] {1f, 2f, 3f, 1f, 2f, 3f}, expanded);
        assertTrue(vector.buffer().data() == expanded.buffer().data());
    }

    @Test
    void permuteReordersAxesWithoutCopy() {
        Tensor cube = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 2, 2, 2);
        Tensor permuted = cube.permute(2, 0, 1);
        assertShape(new int[] {2, 2, 2}, permuted);
        assertAllClose(new float[] {1f, 3f, 5f, 7f, 2f, 4f, 6f, 8f}, permuted);
        assertTrue(cube.buffer().data() == permuted.buffer().data());
    }

    @Test
    void flattenIsReshapeView() {
        Tensor matrix = matrix2d(new float[][] {{1f, 2f}, {3f, 4f}});
        Tensor flat = matrix.flatten();
        assertShape(new int[] {4}, flat);
        assertAllClose(new float[] {1f, 2f, 3f, 4f}, flat);
        flat.set(9f, 0);
        assertClose(9f, matrix.get(0, 0), 1e-5f);
    }

    @Test
    void sumReducesAllElements() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 4);
        assertClose(10f, input.sum().data()[0], 1e-5f);
    }

    @Test
    void sumAxisCollapsesRequestedDimension() {
        Tensor batch = matrix2d(new float[][] {{1f, 2f, 3f}, {4f, 5f, 6f}});
        Tensor reduced = batch.sumAxis(0);
        assertShape(new int[] {3}, reduced);
        assertAllClose(new float[] {5f, 7f, 9f}, reduced);
    }

    @Test
    void expandedBiasCanBeAddedToBatch() {
        Tensor batch = matrix2d(new float[][] {{1f, 2f, 3f}, {4f, 5f, 6f}});
        Tensor bias = Tensor.of(new float[] {10f, 20f, 30f}, 3).expand(2, 3);
        Tensor result = batch.add(bias);
        assertAllClose(new float[] {11f, 22f, 33f, 14f, 25f, 36f}, result);
    }
}
