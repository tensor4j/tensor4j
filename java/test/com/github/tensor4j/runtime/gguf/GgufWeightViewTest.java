/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.github.tensor4j.support.MiniChatGgufBuilder;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

class GgufWeightViewTest {

    @Test
    void sliceDequantMatchesLoadMatrixWithoutTensorBytesCopy() {
        GgufFile file = MiniChatGgufBuilder.buildQ4Model();
        TrackingSource source = new TrackingSource(file);
        float[] viaView = GgufWeightLoader.loadView(source, "token_embd.weight").dequantizeToFloat();
        assertEquals(0, source.tensorBytesCalls);
        float[] viaHeap = GgufWeightLoader.loadMatrix(file, "token_embd.weight").data();
        TensorAssert.assertAllClose(viaHeap, viaView, 0.2f);
    }

    @Test
    void sliceUsesSameMmapBuffer() {
        GgufFile file = MiniChatGgufBuilder.buildIdentityModel();
        GgufTensorSlice slice = file.tensorSlice("output_norm.weight");
        assertSame(file.bytes(), slice.buffer().array());
    }

    private static final class TrackingSource implements GgufTensorSource {

        private final GgufFile delegate;
        private int tensorBytesCalls;

        private TrackingSource(GgufFile delegate) {
            this.delegate = delegate;
        }

        @Override
        public GgufHeader header() {
            return delegate.header();
        }

        @Override
        public byte[] tensorBytes(String name) {
            tensorBytesCalls++;
            return delegate.tensorBytes(name);
        }

        @Override
        public GgufTensorSlice tensorSlice(String name) {
            return delegate.tensorSlice(name);
        }
    }
}
