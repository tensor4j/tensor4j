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

import java.util.HashMap;
import java.util.Map;

/** Fan-out counts on a {@link LazyUOp} DAG (for fusion / scheduler heuristics). */
final class LazyGraphUses {

    private LazyGraphUses() {
    }

    static Map<LazyUOp, Integer> count(LazyUOp root) {
        Map<LazyUOp, Integer> uses = new HashMap<>();
        for (LazyUOp node : LazyGraph.toposort(root)) {
            for (int i = 0; i < node.srcCount(); i++) {
                LazyUOp parent = node.src(i);
                uses.merge(parent, 1, Integer::sum);
            }
        }
        return uses;
    }
}
