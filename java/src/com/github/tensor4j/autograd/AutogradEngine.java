/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.autograd;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import com.github.tensor4j.core.Tensor;

/** Topological backward pass (micrograd-style, single visit per node). */
public final class AutogradEngine {

    private AutogradEngine() {
    }

    public static void backward(Tensor root) {
        backward(root, Tensor.of(1f));
    }

    public static void backward(Tensor root, Tensor seed) {
        List<Tensor> topo = buildTopo(root);
        for (Tensor node : topo) {
            node.zeroGrad();
        }
        root.setGrad(seed);
        for (int index = topo.size() - 1; index >= 0; index--) {
            Tensor node = topo.get(index);
            Tensor grad = node.grad();
            Function creator = node.graphCreator();
            if (grad != null && creator != null) {
                creator.backward(grad);
            }
        }
    }

    private static List<Tensor> buildTopo(Tensor root) {
        List<Tensor> order = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        visit(root, order, visited);
        return order;
    }

    private static void visit(Tensor node, List<Tensor> order, Set<Tensor> visited) {
        if (!visited.add(node)) {
            return;
        }
        Function creator = node.graphCreator();
        if (creator != null) {
            for (Tensor input : creator.inputs()) {
                visit(input, order, visited);
            }
        }
        order.add(node);
    }
}
