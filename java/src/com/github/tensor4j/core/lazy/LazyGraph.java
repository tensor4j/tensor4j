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

import com.github.tensor4j.core.Tensor;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** DAG traversal utilities (tinygrad {@code UOp.toposort}). */
public final class LazyGraph {

    private LazyGraph() {
    }

    /** Dependency order: parents before dependents. Each node appears once even if shared. */
    public static List<LazyUOp> toposort(LazyUOp root) {
        Set<LazyUOp> finished = new LinkedHashSet<>();
        visitToposort(root, finished);
        return new ArrayList<>(finished);
    }

    private static void visitToposort(LazyUOp node, Set<LazyUOp> finished) {
        if (finished.contains(node)) {
            return;
        }
        for (int i = 0; i < node.srcCount(); i++) {
            visitToposort(node.src(i), finished);
        }
        finished.add(node);
    }

    public static int graphDepth(LazyUOp root) {
        if (root.op() == LazyUOp.Kind.BUFFER) {
            return 0;
        }
        int maxChild = 0;
        for (int i = 0; i < root.srcCount(); i++) {
            maxChild = Math.max(maxChild, graphDepth(root.src(i)));
        }
        return 1 + maxChild;
    }

    public static boolean needsGrad(LazyUOp root) {
        for (LazyUOp node : toposort(root)) {
            if (node.op() == LazyUOp.Kind.BUFFER && node.buffer().requiresGrad()) {
                return true;
            }
        }
        return false;
    }

    public static List<Tensor> leafBuffers(LazyUOp root) {
        List<Tensor> leaves = new ArrayList<>();
        Set<LazyUOp> seen = new HashSet<>();
        for (LazyUOp node : toposort(root)) {
            if (node.op() == LazyUOp.Kind.BUFFER && seen.add(node)) {
                leaves.add(node.buffer());
            }
        }
        return leaves;
    }

    public static void zeroGrad(LazyUOp root) {
        Set<Tensor> cleared = new HashSet<>();
        for (LazyUOp node : toposort(root)) {
            if (node.op() == LazyUOp.Kind.BUFFER && cleared.add(node.buffer())) {
                node.buffer().zeroGrad();
            }
        }
    }

    public static int nodeCount(LazyUOp root) {
        return toposort(root).size();
    }

    public static boolean references(LazyUOp root, LazyUOp target) {
        for (LazyUOp node : toposort(root)) {
            if (node == target) {
                return true;
            }
        }
        return false;
    }

    /**
     * Nodes on paths from {@code root} to any {@code target} (tinygrad {@code _deepwalk} subset).
     */
    public static java.util.List<LazyUOp> deepwalk(LazyUOp root, Set<LazyUOp> targets) {
        Set<LazyUOp> onPath = new HashSet<>();
        markPathFromRoot(root, targets, onPath);
        java.util.List<LazyUOp> ordered = new ArrayList<>();
        for (LazyUOp node : toposort(root)) {
            if (onPath.contains(node)) {
                ordered.add(node);
            }
        }
        return ordered;
    }

    private static boolean markPathFromRoot(LazyUOp node, Set<LazyUOp> targets, Set<LazyUOp> onPath) {
        if (targets.contains(node)) {
            onPath.add(node);
            return true;
        }
        boolean reachable = false;
        for (int index = 0; index < node.srcCount(); index++) {
            if (markPathFromRoot(node.src(index), targets, onPath)) {
                reachable = true;
            }
        }
        if (reachable) {
            onPath.add(node);
        }
        return reachable;
    }

    /** Clears intern cache (tests only). */
    public static void clearInternCacheForTests() {
        LazyUOpCache.clearForTests();
    }
}
