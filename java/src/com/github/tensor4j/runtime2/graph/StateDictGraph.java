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

import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Ordered forward graph with deferred {@link com.github.tensor4j.runtime2.state.StateDictWeight} realize. */
public final class StateDictGraph {

    private final List<StateDictNode> nodes;
    private final String outputName;

    public StateDictGraph(List<StateDictNode> nodes, String outputName) {
        this.nodes = List.copyOf(nodes);
        this.outputName = outputName;
    }

    public InferTensor run(Map<String, InferTensor> inputs) {
        Map<String, InferTensor> env = new HashMap<>(inputs);
        for (StateDictNode node : nodes) {
            if (node.op() == StateDictOp.INPUT) {
                if (!env.containsKey(node.name())) {
                    throw new IllegalArgumentException("missing input " + node.name());
                }
                continue;
            }
            env.put(node.name(), eval(node, env));
        }
        InferTensor out = env.get(outputName);
        if (out == null) {
            throw new IllegalStateException("output " + outputName + " not produced");
        }
        return out;
    }

    private static InferTensor eval(StateDictNode node, Map<String, InferTensor> env) {
        InferTensor a = env.get(node.inputA());
        return switch (node.op()) {
            case RMS_NORM -> GgmlOps.rmsNorm(a, node.resolvedWeight(), node.eps());
            case MUL_MAT -> GgmlOps.mulMat(a, node.resolvedWeight());
            case ADD -> GgmlOps.add(a, env.get(node.inputB()));
            case SILU -> GgmlOps.silu(a);
            case SWIGLU -> GgmlOps.swiglu(a, env.get(node.inputB()));
            case MUL -> GgmlOps.mul(a, node.resolvedWeight());
            default -> throw new IllegalStateException("unsupported op " + node.op());
        };
    }
}
