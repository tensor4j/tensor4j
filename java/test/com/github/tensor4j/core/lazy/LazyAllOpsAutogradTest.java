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
import static com.github.tensor4j.support.TensorAssert.matrix2d;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Layer 2o: lazy UOp autograd parity with eager {@link com.github.tensor4j.autograd.AllOpsAutogradTest}.
 */
class LazyAllOpsAutogradTest {

    private static final float EPS = 1e-4f;

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void backwardAdd() {
        assertLazyMatchesEager(
                Tensor.of(new float[] {1f, 2f}, 2).withGrad(true),
                Tensor.of(new float[] {3f, 4f}, 2).withGrad(true),
                new BinaryGraph(OpKind.ADD));
    }

    @Test
    void backwardSub() {
        assertLazyMatchesEager(
                Tensor.of(new float[] {5f, 6f}, 2).withGrad(true),
                Tensor.of(new float[] {1f, 2f}, 2).withGrad(true),
                new BinaryGraph(OpKind.SUB));
    }

    @Test
    void backwardMul() {
        assertLazyMatchesEager(
                Tensor.of(new float[] {2f, 3f}, 2).withGrad(true),
                Tensor.of(new float[] {4f, 5f}, 2).withGrad(true),
                new BinaryGraph(OpKind.MUL));
    }

    @Test
    void backwardDiv() {
        assertLazyMatchesEager(
                Tensor.of(new float[] {6f, 8f}, 2).withGrad(true),
                Tensor.of(new float[] {2f, 4f}, 2).withGrad(true),
                new BinaryGraph(OpKind.DIV));
    }

    @Test
    void backwardNeg() {
        float[] data = new float[] {1f, -2f};
        Tensor eager = Tensor.of(data, 2).withGrad(true);
        eager.neg().sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 2).withGrad(true);
        lazy.neg().sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void backwardRelu() {
        float[] data = new float[] {-1f, 2f, 3f};
        Tensor eager = Tensor.of(data, 3).withGrad(true);
        eager.relu().sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 3).withGrad(true);
        lazy.relu().sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void backwardSum() {
        float[] data = new float[] {1f, 2f, 3f};
        Tensor eager = Tensor.of(data, 3).withGrad(true);
        eager.sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 3).withGrad(true);
        lazy.sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void backwardMean() {
        float[] data = new float[] {2f, 4f, 6f};
        Tensor eager = Tensor.of(data, 3).withGrad(true);
        eager.mean().backward();
        LazyTensor lazy = LazyTensor.of(data, 3).withGrad(true);
        lazy.mean().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void backwardPow2() {
        float[] data = new float[] {2f, 3f};
        Tensor eager = Tensor.of(data, 2).withGrad(true);
        eager.pow2().sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 2).withGrad(true);
        lazy.pow2().sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void backwardReshape() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor eager = Tensor.of(data, 2, 2).withGrad(true);
        eager.reshape(4).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 2, 2).withGrad(true);
        lazy.reshape(4).sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void backwardExpand() {
        float[] data = new float[] {2f, 3f};
        Tensor eager = Tensor.of(data, 2).withGrad(true);
        eager.expand(2, 2).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 2).withGrad(true);
        lazy.expand(2, 2).sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void backwardPermute() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor eager = Tensor.of(data, 2, 2).withGrad(true);
        eager.permute(1, 0).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 2, 2).withGrad(true);
        lazy.permute(1, 0).sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void backwardBiasAdd() {
        float[][] actData = new float[][] {{1f, 2f}, {3f, 4f}};
        float[] biasData = new float[] {0.5f, -0.5f};
        Tensor activations = matrix2d(actData).withGrad(true);
        Tensor bias = Tensor.of(biasData, 2).withGrad(true);
        activations.add(bias.expand(2, 2)).sum().backward();

        LazyTensor lazyAct = LazyTensor.of(flatten(actData), 2, 2).withGrad(true);
        LazyTensor lazyBias = LazyTensor.of(biasData, 2).withGrad(true);
        lazyAct.add(lazyBias.expand(2, 2)).sum().backward();

        assertAllClose(activations.grad(), lazyAct.grad(), EPS);
        assertAllClose(bias.grad(), lazyBias.grad(), EPS);
    }

    @Test
    void fullLazyGraphBackwardNoNaN() {
        float[] inputData = new float[] {0.5f, -1f, 0.25f};
        float[] weightData = new float[] {1f, 2f, -1f};
        float[] biasData = new float[] {0.1f};

        LazyTensor input = LazyTensor.of(inputData, 1, 3).withGrad(true);
        LazyTensor weight = LazyTensor.of(weightData, 3, 1).withGrad(true);
        LazyTensor bias = LazyTensor.of(biasData, 1).withGrad(true);
        LazyTensor hidden = input.matmul(weight).add(bias.expand(1, 1));
        LazyTensor loss = hidden.relu().pow2().mean();
        loss.backward();

        assertNotNull(input.grad());
        assertNotNull(weight.grad());
        assertNotNull(bias.grad());
        assertNoNaN(input.grad());
        assertNoNaN(weight.grad());
        assertNoNaN(bias.grad());
    }

    private static void assertLazyMatchesEager(Tensor eagerLeft, Tensor eagerRight, BinaryGraph graph) {
        graph.runEager(eagerLeft, eagerRight);
        float[] leftData = eagerLeft.toFlatArray();
        float[] rightData = eagerRight.toFlatArray();
        LazyTensor lazyLeft = LazyTensor.of(leftData, eagerLeft.shape().dims()).withGrad(true);
        LazyTensor lazyRight = LazyTensor.of(rightData, eagerRight.shape().dims()).withGrad(true);
        graph.runLazy(lazyLeft, lazyRight);
        assertGradPresent(lazyLeft.leafTensor());
        assertGradPresent(lazyRight.leafTensor());
        assertAllClose(eagerLeft.grad(), lazyLeft.grad(), EPS);
        assertAllClose(eagerRight.grad(), lazyRight.grad(), EPS);
    }

    private static float[] flatten(float[][] matrix) {
        float[] flat = new float[matrix.length * matrix[0].length];
        int index = 0;
        for (float[] row : matrix) {
            for (float value : row) {
                flat[index++] = value;
            }
        }
        return flat;
    }

    private enum OpKind {
        ADD, SUB, MUL, DIV
    }

    private static final class BinaryGraph {
        private final OpKind kind;

        BinaryGraph(OpKind kind) {
            this.kind = kind;
        }

        void runEager(Tensor left, Tensor right) {
            Tensor loss = combine(left, right).sum();
            loss.backward();
        }

        void runLazy(LazyTensor left, LazyTensor right) {
            combine(left, right).sum().backward();
        }

        private Tensor combine(Tensor left, Tensor right) {
            return switch (kind) {
                case ADD -> left.add(right);
                case SUB -> left.sub(right);
                case MUL -> left.mul(right);
                case DIV -> left.div(right);
            };
        }

        private LazyTensor combine(LazyTensor left, LazyTensor right) {
            return switch (kind) {
                case ADD -> left.add(right);
                case SUB -> left.sub(right);
                case MUL -> left.mul(right);
                case DIV -> left.div(right);
            };
        }
    }
}
