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

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Layer 2p: {@link LazyGradient} rule coverage (tinygrad {@code pm_gradient} subset). */
class LazyGradientRulesTest {

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void divBackwardUsesRecipRule() {
        LazyTensor left = LazyTensor.of(new float[] {6f, 8f}, 2).withGrad(true);
        LazyTensor right = LazyTensor.of(new float[] {2f, 4f}, 2).withGrad(true);
        LazyUOp loss = left.div(right).sum().uop();
        var grads = LazyGradient.compute(loss, LazyUOp.buffer(Tensor.of(1f)), LazyGradient.gradTargets(loss));
        assertTrue(hasKind(grads.get(right.uop()), LazyUOp.Kind.RECIP));
    }

    @Test
    void recipBackwardBuildsSquareOfOutput() {
        LazyTensor x = LazyTensor.of(new float[] {2f, 4f}, 2).withGrad(true);
        LazyUOp loss = x.recip().sum().uop();
        var grads = LazyGradient.compute(loss, LazyUOp.buffer(Tensor.of(1f)), LazyGradient.gradTargets(loss));
        assertTrue(hasKind(grads.get(x.uop()), LazyUOp.Kind.MUL));
    }

    @Test
    void meanForwardBuildsMeanNode() {
        LazyTensor x = LazyTensor.of(new float[] {3f, 6f, 9f}, 3).withGrad(true);
        assertTrue(hasKind(x.mean().uop(), LazyUOp.Kind.MEAN));
    }

    @Test
    void maxBackwardBuildsMaskOps() {
        LazyTensor left = LazyTensor.of(new float[] {1f, 5f}, 2).withGrad(true);
        LazyTensor right = LazyTensor.of(new float[] {2f, 4f}, 2).withGrad(true);
        LazyUOp loss = left.max(right).sum().uop();
        var grads = LazyGradient.compute(loss, LazyUOp.buffer(Tensor.of(1f)), LazyGradient.gradTargets(loss));
        assertTrue(hasKind(grads.get(left.uop()), LazyUOp.Kind.GT_MASK));
    }

    @Test
    void padBackwardBuildsShrink() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f}, 2).withGrad(true);
        LazyUOp loss = x.pad(1, 1).sum().uop();
        var grads = LazyGradient.compute(loss, LazyUOp.buffer(Tensor.of(1f)), LazyGradient.gradTargets(loss));
        assertTrue(hasKind(grads.get(x.uop()), LazyUOp.Kind.SHRINK));
    }

    private static boolean hasKind(LazyUOp root, LazyUOp.Kind kind) {
        if (root == null) {
            return false;
        }
        for (LazyUOp node : LazyGraph.toposort(root)) {
            if (node.op() == kind) {
                return true;
            }
        }
        return false;
    }
}
