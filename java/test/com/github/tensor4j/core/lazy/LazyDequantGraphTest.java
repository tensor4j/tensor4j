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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.runtime.ggml.GgmlQuant;
import com.github.tensor4j.runtime2.ggml.reference.TinygradQ4DequantGoldenCase;
import com.github.tensor4j.runtime2.ggml.reference.TinygradQ4DequantGoldenCases;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Lazy UOp GGUF load: shape metadata before realize (tinygrad {@code ggml_data_to_tensor}). */
class LazyDequantGraphTest {

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void shapeInferredWithoutRealize() {
        TinygradQ4DequantGoldenCase golden = TinygradQ4DequantGoldenCases.all()[2];
        LazyGgufSlice slice = LazyGgufSlice.q4_0(golden.quantBytes(), golden.shape());
        LazyTensor weight = LazyTensor.ggufLoad(slice);

        assertFalse(weight.isRealized());
        assertEquals(LazyUOp.Kind.DEQUANT_Q4_0, weight.uop().op());
        assertSame(slice, weight.uop().ggufSlice());
        assertArrayEquals(golden.shape(), weight.shape());
        assertArrayEquals(slice.q4BlockShape(), new int[] {2, LazyQuantMath.BLOCK_BYTES_Q4_0});
    }

    @Test
    void f32MmapLazyLoad() {
        float[] source = new float[] {1f, 2f, 3f, 4f};
        byte[] bytes = new byte[16];
        for (int i = 0; i < source.length; i++) {
            bytes[i * 4] = (byte) (Float.floatToRawIntBits(source[i]));
            bytes[i * 4 + 1] = (byte) (Float.floatToRawIntBits(source[i]) >> 8);
            bytes[i * 4 + 2] = (byte) (Float.floatToRawIntBits(source[i]) >> 16);
            bytes[i * 4 + 3] = (byte) (Float.floatToRawIntBits(source[i]) >> 24);
        }
        LazyGgufSlice slice = LazyGgufSlice.f32(LazyQuantMath.wrapBytes(bytes), 0, new int[] {4});
        LazyTensor lazy = LazyTensor.ggufLoad(slice);
        assertEquals(LazyUOp.Kind.MMAP_F32, lazy.uop().op());
        assertFalse(lazy.isRealized());
        TensorAssert.assertAllClose(source, lazy.realize().data(), 1e-6f);
    }

    @Test
    void matmulDefersLoadUntilRealize() {
        TinygradQ4DequantGoldenCase golden = TinygradQ4DequantGoldenCases.all()[2];
        LazyGgufSlice slice = LazyGgufSlice.q4_0(golden.quantBytes(), golden.shape());
        int rows = golden.shape()[0];
        int cols = golden.shape()[1];
        LazyTensor w = LazyTensor.ggufLoad(slice);
        LazyTensor x = LazyTensor.of(new float[rows], 1, rows);
        LazyTensor y = x.dot(w);

        assertFalse(w.isRealized());
        assertFalse(y.isRealized());
        y.realize();
        assertTrue(y.isRealized());
        w.realize();
        assertTrue(w.isRealized());
    }

    @Test
    void realizeMatchesGgmlQuantReference() {
        for (TinygradQ4DequantGoldenCase golden : TinygradQ4DequantGoldenCases.all()) {
            LazyGgufSlice slice = LazyGgufSlice.q4_0(golden.quantBytes(), golden.shape());
            float[] viaLazy = LazyTensor.ggufLoad(slice).realize().data();
            float[] viaGgml = GgmlQuant.dequantizeQ4_0(
                    golden.quantBytes(),
                    TinygradQ4DequantGoldenCases.toShape(golden.shape()));
            TensorAssert.assertAllClose(viaGgml, viaLazy, golden.tolerance());
        }
    }
}
