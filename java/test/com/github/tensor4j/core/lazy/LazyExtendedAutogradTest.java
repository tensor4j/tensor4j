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

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static com.github.tensor4j.support.TensorAssert.assertGradPresent;
import static com.github.tensor4j.support.TensorAssert.assertNoNaN;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Layer 2r: autograd for extended lazy ops (tinygrad pm_gradient subset). */
class LazyExtendedAutogradTest {

    private static final float EPS = 5e-2f;
    private static final float FD_EPS = 1e-3f;

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void backwardPow() {
        assertLazyGradClose(new float[] {2f, 3f}, 2, x -> x.pow(2f));
    }

    @Test
    void backwardSqrt() {
        assertLazyGradClose(new float[] {4f, 9f, 16f}, 3, LazyTensor::sqrt);
    }

    @Test
    void backwardLog2() {
        assertLazyGradClose(new float[] {2f, 4f, 8f}, 3, LazyTensor::log2);
    }

    @Test
    void backwardExp2() {
        assertLazyGradClose(new float[] {1f, 2f, 3f}, 3, LazyTensor::exp2);
    }

    @Test
    void backwardMax() {
        LazyTensor left = LazyTensor.of(new float[] {1f, 5f}, 2).withGrad(true);
        LazyTensor right = LazyTensor.of(new float[] {2f, 4f}, 2).withGrad(true);
        left.max(right).sum().backward();
        assertGradPresent(left.leafTensor());
        assertGradPresent(right.leafTensor());
        assertNoNaN(left.grad());
        assertNoNaN(right.grad());
        assertAllClose(new float[] {0f, 1f}, left.grad(), EPS);
        assertAllClose(new float[] {1f, 0f}, right.grad(), EPS);
    }

    @Test
    void backwardWhere() {
        LazyTensor cond = LazyTensor.of(new float[] {1f, 0f}, 2);
        LazyTensor ifTrue = LazyTensor.of(new float[] {3f, 4f}, 2).withGrad(true);
        LazyTensor ifFalse = LazyTensor.of(new float[] {5f, 6f}, 2).withGrad(true);
        cond.where(ifTrue, ifFalse).sum().backward();
        assertAllClose(new float[] {1f, 0f}, ifTrue.grad(), EPS);
        assertAllClose(new float[] {0f, 1f}, ifFalse.grad(), EPS);
    }

    @Test
    void backwardPadShrink() {
        LazyTensor x = LazyTensor.of(new float[] {2f, 3f}, 2).withGrad(true);
        x.pad(1, 1).shrink(1, 3).sum().backward();
        assertAllClose(new float[] {1f, 1f}, x.grad(), EPS);
    }

    @Test
    void backwardContiguousCastPassthrough() {
        LazyTensor x = LazyTensor.of(new float[] {2f, 4f}, 2).withGrad(true);
        x.contiguous().cast().sum().backward();
        assertAllClose(new float[] {1f, 1f}, x.grad(), EPS);
    }

    @Test
    void powGradientRuleUsesPowNode() {
        LazyTensor x = LazyTensor.of(new float[] {2f}, 1).withGrad(true);
        LazyUOp loss = x.pow(3f).sum().uop();
        var grads = LazyGradient.compute(loss, LazyUOp.buffer(Tensor.of(1f)), LazyGradient.gradTargets(loss));
        assertTrue(hasKind(grads.get(x.uop()), LazyUOp.Kind.POW));
    }

    private static void assertLazyGradClose(float[] base, int length, java.util.function.Function<LazyTensor, LazyTensor> graph) {
        int[] shape = new int[] {length};
        LazyTensor param = LazyTensor.of(base.clone(), shape).withGrad(true);
        graph.apply(param).sum().backward();
        assertGradPresent(param.leafTensor());
        float[] analytical = param.grad().data();

        float[] scratch = base.clone();
        for (int index = 0; index < base.length; index++) {
            scratch[index] = base[index] + FD_EPS;
            LazyTensor plusParam = LazyTensor.of(scratch, shape).withGrad(true);
            float plus = graph.apply(plusParam).sum().realize().data()[0];

            scratch[index] = base[index] - FD_EPS;
            LazyTensor minusParam = LazyTensor.of(scratch, shape).withGrad(true);
            float minus = graph.apply(minusParam).sum().realize().data()[0];

            scratch[index] = base[index];
            float numeric = (plus - minus) / (2f * FD_EPS);
            assertAllClose(new float[] {numeric}, Tensor.of(analytical[index]), EPS);
        }
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
