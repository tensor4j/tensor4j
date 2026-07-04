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

import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Ordered forward graph executor (ggml cgraph compute, no autograd). */
public final class InferGraph {

    private final List<InferNode> nodes;
    private final String outputName;

    public InferGraph(List<InferNode> nodes, String outputName) {
        this.nodes = List.copyOf(nodes);
        this.outputName = outputName;
    }

    public InferTensor run(Map<String, InferTensor> inputs) {
        Map<String, InferTensor> env = new HashMap<>(inputs);
        for (InferNode node : nodes) {
            if (node.op() == InferOp.INPUT) {
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

    private static InferTensor eval(InferNode node, Map<String, InferTensor> env) {
        InferTensor a = env.get(node.inputA());
        switch (node.op()) {
            case RMS_NORM -> {
                return GgmlOps.rmsNorm(a, node.weight(), node.eps());
            }
            case MUL_MAT -> {
                return GgmlOps.mulMat(a, node.weight());
            }
            case ADD -> {
                return GgmlOps.add(a, env.get(node.inputB()));
            }
            case SILU -> {
                return GgmlOps.silu(a);
            }
            case SWIGLU -> {
                return GgmlOps.swiglu(a, env.get(node.inputB()));
            }
            case MUL -> {
                return GgmlOps.mul(a, node.weight());
            }
            default -> throw new IllegalStateException("unsupported op " + node.op());
        }
    }
}
