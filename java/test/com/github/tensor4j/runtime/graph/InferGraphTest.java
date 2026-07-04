/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.support.TensorAssert;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class InferGraphTest {

    @Test
    void ffnSubgraphMatchesEager() {
        float eps = 1e-5f;
        InferTensor x = InferTensor.vector(1, 2);
        InferTensor normW = InferTensor.vector(1, 1);
        InferTensor w1 = InferTensor.matrix(2, 2, 1, 0, 0, 1);
        InferTensor w2 = InferTensor.matrix(2, 2, 0, 1, 1, 0);
        List<InferNode> nodes = List.of(
                InferNode.input("x"),
                InferNode.rmsNorm("n", "x", normW, eps),
                InferNode.mulMat("g", "n", w1),
                InferNode.mulMat("u", "n", w2),
                InferNode.swiglu("m", "g", "u"),
                InferNode.add("y", "x", "m"));
        InferGraph graph = new InferGraph(nodes, "y");
        InferTensor out = graph.run(Map.of("x", x));

        InferTensor eager = GgmlOps.add(x, GgmlOps.swiglu(
                GgmlOps.mulMat(GgmlOps.rmsNorm(x, normW, eps), w1),
                GgmlOps.mulMat(GgmlOps.rmsNorm(x, normW, eps), w2)));
        TensorAssert.assertAllClose(eager.data(), out.data(), 1e-4f);
    }

    @Test
    void outputNameResolved() {
        InferGraph graph = new InferGraph(List.of(
                InferNode.input("x"),
                InferNode.silu("y", "x")), "y");
        InferTensor x = InferTensor.vector(-1, 0, 1);
        assertEquals(GgmlOps.silu(x).get(0, 1), graph.run(Map.of("x", x)).get(0, 1), 1e-5f);
    }
}
