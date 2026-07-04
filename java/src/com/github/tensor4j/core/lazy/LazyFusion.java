/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core.lazy;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Fuses elementwise {@link LazyUOp} chains into kernels (tinygrad kernel fusion subset).
 * Movement / reduce ops remain separate schedule entries.
 */
final class LazyFusion {

    private LazyFusion() {
    }

    static boolean isElementwise(LazyUOp.Kind op) {
        return op == LazyUOp.Kind.ADD || op == LazyUOp.Kind.MUL || op == LazyUOp.Kind.SUB || op == LazyUOp.Kind.DIV
                || op == LazyUOp.Kind.NEG || op == LazyUOp.Kind.RECIP || op == LazyUOp.Kind.RELU
                || op == LazyUOp.Kind.POW || op == LazyUOp.Kind.LOG2 || op == LazyUOp.Kind.EXP2
                || op == LazyUOp.Kind.EXP || op == LazyUOp.Kind.SQRT || op == LazyUOp.Kind.MAX
                || op == LazyUOp.Kind.GT_MASK || op == LazyUOp.Kind.EQ_MASK;
    }

    static boolean isMovement(LazyUOp.Kind op) {
        return op == LazyUOp.Kind.RESHAPE || op == LazyUOp.Kind.PERMUTE || op == LazyUOp.Kind.EXPAND
                || op == LazyUOp.Kind.PAD || op == LazyUOp.Kind.SHRINK
                || op == LazyUOp.Kind.CONTIGUOUS || op == LazyUOp.Kind.CAST;
    }

    static boolean isReduce(LazyUOp.Kind op) {
        return op == LazyUOp.Kind.SUM || op == LazyUOp.Kind.SUM_AXIS || op == LazyUOp.Kind.MAX_AXIS
                || op == LazyUOp.Kind.MEAN || op == LazyUOp.Kind.IM2COL;
    }

    /** Grow a maximal fusible elementwise subgraph ending at {@code sink}. */
    static LazyKernel fusedKernel(LazyUOp sink, Map<LazyUOp, Integer> uses, Set<LazyUOp> claimed) {
        Set<LazyUOp> body = new LinkedHashSet<>();
        collectFusible(sink, uses, body);
        if (body.size() <= 1) {
            return LazyKernel.single(sink);
        }
        List<LazyUOp> ordered = new ArrayList<>();
        for (LazyUOp node : LazyGraph.toposort(sink)) {
            if (body.contains(node)) {
                ordered.add(node);
            }
        }
        claimed.addAll(body);
        return LazyKernel.fused(ordered);
    }

    private static void collectFusible(LazyUOp node, Map<LazyUOp, Integer> uses, Set<LazyUOp> body) {
        if (!body.add(node)) {
            return;
        }
        if (node.op() == LazyUOp.Kind.BUFFER) {
            return;
        }
        if (isMovement(node.op()) || isReduce(node.op())) {
            return;
        }
        if (!isElementwise(node.op())) {
            return;
        }
        for (int i = 0; i < node.srcCount(); i++) {
            LazyUOp parent = node.src(i);
            int fanout = uses.getOrDefault(parent, 0);
            if (fanout <= 1 && (parent.op() == LazyUOp.Kind.BUFFER || isElementwise(parent.op()))) {
                collectFusible(parent, uses, body);
            }
        }
    }

    static List<LazyKernel> withoutFusion(LazyUOp root) {
        List<LazyKernel> kernels = new ArrayList<>();
        for (LazyUOp node : LazyGraph.toposort(root)) {
            if (node.op() != LazyUOp.Kind.BUFFER) {
                kernels.add(LazyKernel.single(node));
            }
        }
        return kernels;
    }
}
