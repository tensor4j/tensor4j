/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime2.state;

import com.github.tensor4j.core.lazy.LazyTensor;
import com.github.tensor4j.runtime.gguf.GgufWeightView;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.runtime2.lazy.GgufLazyWeightGraph;

/**
 * One entry from a tinygrad-style state dict backed by a GGUF mmap slice.
 *
 * <p>{@link #lazyTensor()} is the canonical UOp graph before realize;
 * {@link #tensor()} realizes it ({@code load_state_dict(..., realize=True)}).
 */
public final class StateDictWeight {

    public enum Layout {
        MATRIX,
        EMBEDDING,
        VECTOR
    }

    /** When to materialize float weights from the lazy UOp graph. */
    public enum RealizeMode {
        /** Keep mmap + lazy UOp until {@link #tensor()} or {@link LazyTensor#realize()}. */
        ON_DEMAND,
        /** Call {@link #tensor()} immediately at load (tinygrad {@code realize=True}). */
        AT_LOAD
    }

    private final GgufWeightView view;
    private final Layout layout;
    private LazyTensor lazyGraph;
    private InferTensor cached;

    private StateDictWeight(GgufWeightView view, Layout layout) {
        this.view = view;
        this.layout = layout;
    }

    public static StateDictWeight from(GgufWeightView view, Layout layout) {
        return new StateDictWeight(view, layout);
    }

    public static StateDictWeight from(GgufWeightView view, Layout layout, RealizeMode mode) {
        StateDictWeight weight = from(view, layout);
        if (mode == RealizeMode.AT_LOAD) {
            weight.tensor();
        }
        return weight;
    }

    public GgufWeightView view() {
        return view;
    }

    /**
     * Canonical lazy UOp graph (tinygrad {@code state_dict[name]} before realize).
     * Cached — same instance on repeat calls.
     */
    public LazyTensor lazyTensor() {
        if (lazyGraph == null) {
            lazyGraph = GgufLazyWeightGraph.build(view, layout);
        }
        return lazyGraph;
    }

    /** Realize lazy graph into row-major {@link InferTensor}. */
    public InferTensor tensor() {
        if (cached == null) {
            cached = GgufLazyWeightGraph.toInferTensor(lazyTensor().realize(), layout);
        }
        return cached;
    }
}
