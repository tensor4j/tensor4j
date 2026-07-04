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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.lazy.LazyTensor;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufFile;
import com.github.tensor4j.runtime.gguf.GgufWeightLoader;
import com.github.tensor4j.runtime2.lazy.GgufLazyWeightGraph;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

/** Unified {@link StateDictWeight} load: one lazy UOp graph, {@link StateDictWeight#tensor()} = realize. */
class StateDictWeightTest {

    @Test
    void lazyTensorCachedAndShapeBeforeRealize() {
        GgufFile file = MiniChatGgufBuilder.buildQ4Model();
        StateDictWeight weight = StateDictWeight.from(
                GgufWeightLoader.loadView(file, "blk.0.attn_q.weight"),
                StateDictWeight.Layout.MATRIX);

        LazyTensor first = weight.lazyTensor();
        LazyTensor second = weight.lazyTensor();
        assertSame(first, second);
        assertFalse(first.isRealized());
        assertArrayEquals(
                new int[] {MiniChatGgufBuilder.N_EMBD, MiniChatGgufBuilder.N_EMBD},
                first.shape());
    }

    @Test
    void tensorRealizesUnifiedLazyGraph() {
        GgufFile q4 = MiniChatGgufBuilder.buildQ4Model();
        GgufFile f32 = MiniChatGgufBuilder.buildIdentityModel();
        StateDictWeight q4Weight = StateDictWeight.from(
                GgufWeightLoader.loadView(q4, "blk.0.ffn_up.weight"),
                StateDictWeight.Layout.MATRIX);
        StateDictWeight f32Weight = StateDictWeight.from(
                GgufWeightLoader.loadView(f32, "blk.0.ffn_up.weight"),
                StateDictWeight.Layout.MATRIX);

        assertFalse(q4Weight.lazyTensor().isRealized());
        TensorAssert.assertAllClose(
                GgufLazyWeightGraph.materializeDirect(q4Weight.view(), StateDictWeight.Layout.MATRIX).data(),
                q4Weight.tensor().data(),
                0.2f);
        TensorAssert.assertAllClose(
                GgufLazyWeightGraph.materializeDirect(f32Weight.view(), StateDictWeight.Layout.MATRIX).data(),
                f32Weight.tensor().data(),
                1e-5f);
        assertTrue(q4Weight.lazyTensor().isRealized());
    }

    @Test
    void f32LazyGraphBeforeRealize() {
        GgufFile file = MiniChatGgufBuilder.buildIdentityModel();
        StateDictWeight weight = StateDictWeight.from(
                GgufWeightLoader.loadView(file, "output_norm.weight"),
                StateDictWeight.Layout.VECTOR);
        assertFalse(weight.lazyTensor().isRealized());
        TensorAssert.assertAllClose(
                weight.tensor().data(),
                GgufLazyWeightGraph.materializeDirect(weight.view(), StateDictWeight.Layout.VECTOR).data(),
                1e-5f);
    }

    @Test
    void atLoadRealizeModeMaterializesImmediately() {
        GgufFile file = MiniChatGgufBuilder.buildQ4Model();
        StateDictWeight onDemand = StateDictWeight.from(
                GgufWeightLoader.loadView(file, "blk.0.attn_k.weight"),
                StateDictWeight.Layout.MATRIX);
        StateDictWeight atLoad = StateDictWeight.from(
                GgufWeightLoader.loadView(file, "blk.0.attn_k.weight"),
                StateDictWeight.Layout.MATRIX,
                StateDictWeight.RealizeMode.AT_LOAD);

        assertFalse(onDemand.lazyTensor().isRealized());
        assertTrue(atLoad.lazyTensor().isRealized());
        TensorAssert.assertAllClose(onDemand.tensor().data(), atLoad.tensor().data(), 1e-5f);
    }
}
