/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime2.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufFile;
import com.github.tensor4j.runtime.gguf.GgufWeightLoader;
import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.runtime2.state.StateDictWeight;
import com.github.tensor4j.support.TensorAssert;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** {@link StateDictGraph} defers weight dequant until node eval (tinygrad lazy realize). */
class StateDictGraphTest {

    private static final float EPS = 1e-5f;

    @Test
    void ffnSubgraphWithLazyQ4WeightsMatchesEager() {
        GgufFile file = MiniChatGgufBuilder.buildQ4Model();
        StateDictWeight w1 = StateDictWeight.from(
                GgufWeightLoader.loadView(file, "blk.0.ffn_up.weight"),
                StateDictWeight.Layout.MATRIX);
        InferTensor w2 = GgufWeightLoader.loadMatrix(file, "blk.0.ffn_gate.weight");

        InferTensor x = smokeInput(MiniChatGgufBuilder.N_EMBD);
        InferTensor normW = GgufWeightLoader.loadVector(file, "blk.0.ffn_norm.weight");

        List<StateDictNode> nodes = List.of(
                StateDictNode.input("x"),
                StateDictNode.rmsNorm("n", "x", normW, EPS),
                StateDictNode.mulMat("g", "n", w2),
                StateDictNode.mulMat("u", "n", w1),
                StateDictNode.swiglu("m", "g", "u"),
                StateDictNode.add("y", "x", "m"));
        StateDictGraph graph = new StateDictGraph(nodes, "y");
        InferTensor out = graph.run(Map.of("x", x));

        InferTensor eager = GgmlOps.add(x, GgmlOps.swiglu(
                GgmlOps.mulMat(GgmlOps.rmsNorm(x, normW, EPS), w2),
                GgmlOps.mulMat(GgmlOps.rmsNorm(x, normW, EPS), w1.tensor())));
        TensorAssert.assertAllClose(eager.data(), out.data(), 0.25f);
    }

    @Test
    void lazyWeightMaterializesOnce() {
        GgufFile file = MiniChatGgufBuilder.buildQ4Model();
        StateDictWeight lazy = StateDictWeight.from(
                GgufWeightLoader.loadView(file, "blk.0.attn_q.weight"),
                StateDictWeight.Layout.MATRIX);
        InferTensor first = lazy.tensor();
        InferTensor second = lazy.tensor();
        assertEquals(first, second);
        assertEquals(MiniChatGgufBuilder.N_EMBD, first.rows());
        assertEquals(MiniChatGgufBuilder.N_EMBD, first.cols());
        assertTrue(lazy.lazyTensor().isRealized());
    }

    private static InferTensor smokeInput(int nEmbd) {
        float[] data = new float[nEmbd];
        for (int i = 0; i < nEmbd; i++) {
            data[i] = (i + 1) * 0.1f;
        }
        return InferTensor.of(data, 1, nEmbd);
    }
}
