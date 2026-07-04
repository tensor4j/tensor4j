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

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static com.github.tensor4j.support.TensorAssert.assertClose;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Layer 1c: flat float buffer layout invariants (tinygrad row-major / stride semantics).
 * These tests detect float[][]-style layout bugs without needing Python at runtime.
 */
class FloatLayoutTest {

    @Test
    void flatIndexMatchesManualRowMajorUnravel() {
        Tensor matrix = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
        for (int flat = 0; flat < matrix.numel(); flat++) {
            int[] indices = Strides.unravel(flat, matrix.shape().dims());
            float expected = matrix.buffer().data()[matrix.buffer().offset() + flat];
            assertClose(expected, matrix.get(indices[0], indices[1]), 1e-6f,
                    "get vs contiguous slot at flat " + flat);
            assertClose(expected, matrix.getFlat(flat), 1e-6f, "getFlat at " + flat);
        }
    }

    @Test
    void transposeViewReadsTransposedIndicesWithoutCopy() {
        Tensor source = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2);
        Tensor view = source.transpose2d();
        assertFalse(view.isContiguous());
        assertArrayEquals(new int[] {1, 2}, view.strides());
        assertClose(3f, view.get(0, 1), 1e-6f);
        assertClose(3f, source.get(1, 0), 1e-6f);
        assertAllClose(new float[] {1f, 3f, 2f, 4f}, view.toFlatArray(), 1e-6f);
    }

    @Test
    void expandStrideZeroRepeatsUnderlyingValues() {
        Tensor vector = Tensor.of(new float[] {2f, 4f, 6f}, 3);
        Tensor expanded = vector.expand(3, 3);
        assertArrayEquals(new int[] {0, 1}, expanded.strides());
        for (int row = 0; row < 3; row++) {
            assertClose(2f, expanded.get(row, 0), 1e-6f);
            assertClose(4f, expanded.get(row, 1), 1e-6f);
            assertClose(6f, expanded.get(row, 2), 1e-6f);
        }
        assertAllClose(new float[] {2f, 4f, 6f, 2f, 4f, 6f, 2f, 4f, 6f}, expanded);
    }

    @Test
    void contiguousMaterializesExpandedLayoutToOwnedBuffer() {
        Tensor expanded = Tensor.of(new float[] {1f, 2f}, 2).expand(2, 2);
        Tensor dense = expanded.contiguous();
        assertTrue(dense.isContiguous());
        assertTrue(dense.buffer().isOwnedRoot());
        assertFalse(expanded.buffer() == dense.buffer());
        assertAllClose(new float[] {1f, 2f, 1f, 2f}, dense);
    }

    @Test
    void toFlatArrayMatchesTinygradRowMajorOrder() {
        Tensor cube = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 2, 2, 2);
        assertAllClose(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, cube.toFlatArray(), 1e-6f);
        Tensor permuted = cube.permute(2, 0, 1);
        assertAllClose(new float[] {1f, 3f, 5f, 7f, 2f, 4f, 6f, 8f}, permuted.toFlatArray(), 1e-6f);
    }

    @Test
    void ownedBufferLengthEqualsNumelNotViewSpan() {
        Tensor owned = Tensor.of(new float[] {1f, 2f, 3f}, 3);
        assertEquals(3, owned.buffer().data().length);
        assertEquals(3, owned.numel());
        assertEquals(0, owned.buffer().offset());
    }
}
