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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Linearized kernel list for a lazy DAG (tinygrad {@code create_schedule} + {@code LINEAR}).
 */
public final class LazySchedule {

    private final LazyUOp sink;
    private final List<LazyKernel> kernels;
    private final int unfusedKernelCount;

    private LazySchedule(LazyUOp sink, List<LazyKernel> kernels, int unfusedKernelCount) {
        this.sink = sink;
        this.kernels = List.copyOf(kernels);
        this.unfusedKernelCount = unfusedKernelCount;
    }

    public static LazySchedule build(LazyUOp root) {
        Map<LazyUOp, Integer> uses = LazyGraphUses.count(root);
        Set<LazyUOp> claimed = new HashSet<>();
        List<LazyKernel> kernels = new ArrayList<>();
        List<LazyUOp> forward = LazyGraph.toposort(root);
        List<LazyUOp> outputsFirst = new ArrayList<>(forward);
        java.util.Collections.reverse(outputsFirst);
        for (LazyUOp node : outputsFirst) {
            if (claimed.contains(node) || node.op() == LazyUOp.Kind.BUFFER) {
                continue;
            }
            LazyKernel kernel = LazyFusion.fusedKernel(node, uses, claimed);
            kernels.add(kernel);
            if (kernel.kind() == LazyKernel.Kind.SINGLE) {
                claimed.add(node);
            }
        }
        Map<LazyUOp, Integer> orderIndex = new HashMap<>();
        for (int i = 0; i < forward.size(); i++) {
            orderIndex.put(forward.get(i), i);
        }
        kernels.sort(Comparator.comparingInt(kernel -> orderIndex.get(kernel.output())));
        int unfused = LazyFusion.withoutFusion(root).size();
        return new LazySchedule(root, kernels, unfused);
    }

    public LazyUOp sink() {
        return sink;
    }

    public List<LazyKernel> kernels() {
        return kernels;
    }

    public int kernelCount() {
        return kernels.size();
    }

    public int unfusedKernelCount() {
        return unfusedKernelCount;
    }

    public int savedKernels() {
        return Math.max(0, unfusedKernelCount - kernels.size());
    }

    /** Execute schedule without autograd (fused elementwise loops). */
    public Tensor execute() {
        return LazyScheduleExecutor.execute(sink, kernels);
    }
}
