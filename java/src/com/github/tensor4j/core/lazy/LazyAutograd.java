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
import java.util.Map;
import java.util.Set;

/** Runs tinygrad-style UOp backward and materializes leaf gradients. */
final class LazyAutograd {

    private LazyAutograd() {
    }

    static void backward(LazyUOp root, LazyUOp seed) {
        LazyGraph.zeroGrad(root);
        Set<LazyUOp> targets = LazyGradient.gradTargets(root);
        if (targets.isEmpty()) {
            return;
        }
        Map<LazyUOp, LazyUOp> gradNodes = LazyGradient.compute(root, seed, targets);
        for (LazyUOp target : targets) {
            LazyUOp gradNode = gradNodes.get(target);
            Tensor gradValue;
            if (gradNode == null) {
                gradValue = Tensor.zeros(target.buffer().shape().dims());
            } else {
                gradValue = LazyRealizer.realize(gradNode, false);
            }
            target.buffer().setGrad(gradValue);
        }
    }
}
