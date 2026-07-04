/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime2.lazy;

import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.core.lazy.LazyTensor;
import com.github.tensor4j.runtime.gguf.GgufWeightView;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.runtime2.state.StateDictWeight;

/**
 * Builds lazy UOp graphs from GGUF views (tinygrad {@code ggml_data_to_tensor} + {@code reshape}).
 * Single pipeline for both {@link StateDictWeight#lazyTensor()} and {@link StateDictWeight#tensor()}.
 */
public final class GgufLazyWeightGraph {

    private GgufLazyWeightGraph() {
    }

    public static LazyTensor build(GgufWeightView view, StateDictWeight.Layout layout) {
        LazyTensor flat = LazyTensor.ggufLoad(GgufLazySlice.from(view.slice()));
        return applyLayout(flat, layout, GgufLazySlice.floatDims(view.slice().shape()));
    }

    public static InferTensor toInferTensor(Tensor realized, StateDictWeight.Layout layout) {
        int[] shape = realized.shape().dims();
        float[] data = realized.data();
        return switch (layout) {
            case MATRIX -> {
                if (shape.length != 2) {
                    throw new IllegalStateException("matrix layout expected rank 2, got " + shape.length);
                }
                yield InferTensor.of(data, shape[0], shape[1]);
            }
            case VECTOR -> {
                if (shape.length == 1) {
                    yield InferTensor.vector(data);
                }
                yield InferTensor.of(data, shape[0], shape[1]);
            }
            case EMBEDDING -> {
                if (shape.length != 2) {
                    throw new IllegalStateException("embedding layout expected rank 2, got " + shape.length);
                }
                yield InferTensor.of(data, shape[0], shape[1]);
            }
        };
    }

    /** Eager path via unified lazy graph (for parity checks). */
    public static InferTensor materialize(GgufWeightView view, StateDictWeight.Layout layout) {
        return toInferTensor(build(view, layout).realize(), layout);
    }

    /**
     * llama-track eager dequant without UOp (view → float[] copy).
     * Public for testability — parity baseline vs {@link #materialize(GgufWeightView, StateDictWeight.Layout)}.
     */
    public static InferTensor materializeDirect(GgufWeightView view, StateDictWeight.Layout layout) {
        return switch (layout) {
            case MATRIX -> view.toMatrix();
            case EMBEDDING -> view.toEmbedding();
            case VECTOR -> view.toVector();
        };
    }

    private static LazyTensor applyLayout(LazyTensor flat, StateDictWeight.Layout layout, int[] ggmlDims) {
        return switch (layout) {
            case MATRIX -> flat.reshape(ggmlDims[0], ggmlDims[1]);
            case VECTOR -> flat.reshape(ggmlDims[0]);
            case EMBEDDING -> flat.reshape(ggmlDims[0], ggmlDims[1]).permute(1, 0);
        };
    }
}
