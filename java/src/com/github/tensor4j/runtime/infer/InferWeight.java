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

import com.github.tensor4j.runtime.gguf.GgufWeightView;

/** Lazy or eager inference weight; lazy path dequants from mmap on first {@link #tensor()}. */
public final class InferWeight {

    public enum Layout {
        MATRIX,
        EMBEDDING,
        VECTOR
    }

    private final GgufWeightView view;
    private final Layout layout;
    private final InferTensor eager;
    private InferTensor cached;

    private InferWeight(GgufWeightView view, Layout layout, InferTensor eager) {
        this.view = view;
        this.layout = layout;
        this.eager = eager;
    }

    public static InferWeight lazy(GgufWeightView view, Layout layout) {
        return new InferWeight(view, layout, null);
    }

    public static InferWeight eager(InferTensor tensor) {
        return new InferWeight(null, null, tensor);
    }

    public boolean isLazy() {
        return eager == null;
    }

    public GgufWeightView view() {
        return view;
    }

    public InferTensor tensor() {
        if (eager != null) {
            return eager;
        }
        if (cached == null) {
            if (layout == Layout.MATRIX) {
                cached = view.toMatrix();
            } else if (layout == Layout.EMBEDDING) {
                cached = view.toEmbedding();
            } else {
                cached = view.toVector();
            }
        }
        return cached;
    }
}
