/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime2.ggml;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.runtime.ggml.GgmlLayout;
import com.github.tensor4j.runtime.ggml.GgmlQuant;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import com.github.tensor4j.runtime2.ggml.reference.TinygradQ4DequantGoldenCase;
import com.github.tensor4j.runtime2.ggml.reference.TinygradQ4DequantGoldenCases;
import com.github.tensor4j.runtime2.ggml.reference.TinygradQ4DequantReference;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

/** Q4_0 dequant parity: {@link GgmlQuant} vs tinygrad {@code ggml_data_to_tensor} reference. */
class TinygradQ4DequantParityTest {

    @Test
    void quantByteCountMatchesTinygradBlockLayout() {
        GgmlTensorShape shape = GgmlTensorShape.of(32, 2);
        long elements = shape.numElements();
        assertEquals(64, elements);
        assertEquals(2, elements / GgmlQuant.QK4_0);
        assertEquals(36, GgmlLayout.numBytes(GgmlType.Q4_0, shape));
        assertEquals(36, TinygradQ4DequantReference.quantBytes(shape));
    }

    @Test
    void ggmlQuantMatchesTinygradReferenceOnGoldenCases() {
        for (TinygradQ4DequantGoldenCase golden : TinygradQ4DequantGoldenCases.all()) {
            GgmlTensorShape shape = TinygradQ4DequantGoldenCases.toShape(golden.shape());
            float[] viaGgml = GgmlQuant.dequantizeQ4_0(golden.quantBytes(), shape);
            float[] viaRef = TinygradQ4DequantReference.dequantize(golden.quantBytes(), shape);
            TensorAssert.assertAllClose(viaRef, viaGgml, golden.tolerance());
            TensorAssert.assertAllClose(golden.expected(), viaGgml, golden.tolerance());
        }
    }

    @Test
    void mmapSliceDequantMatchesReference() {
        TinygradQ4DequantGoldenCase golden = TinygradQ4DequantGoldenCases.all()[2];
        GgmlTensorShape shape = TinygradQ4DequantGoldenCases.toShape(golden.shape());
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(golden.quantBytes())
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] viaMmap = GgmlQuant.dequantizeQ4_0(buffer, 0, shape);
        float[] viaRef = TinygradQ4DequantReference.dequantize(golden.quantBytes(), shape);
        TensorAssert.assertAllClose(viaRef, viaMmap, golden.tolerance());
    }
}
