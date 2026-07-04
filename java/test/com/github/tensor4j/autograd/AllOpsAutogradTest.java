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

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static com.github.tensor4j.support.TensorAssert.assertGradPresent;
import static com.github.tensor4j.support.TensorAssert.assertNoNaN;
import static com.github.tensor4j.support.TensorAssert.matrix2d;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.support.GradCheck;
import org.junit.jupiter.api.Test;

/**
 * Layer 2: autograd for all tensor4j ops (tinygrad backward semantics on Java floats).
 */
class AllOpsAutogradTest {

    private static final float EPS = 1e-4f;

    @Test
    void backwardAdd() {
        Tensor left = Tensor.of(new float[] {1f, 2f}, 2).withGrad(true);
        Tensor right = Tensor.of(new float[] {3f, 4f}, 2).withGrad(true);
        left.add(right).sum().backward();
        assertAllClose(new float[] {1f, 1f}, left.grad(), EPS);
        assertAllClose(new float[] {1f, 1f}, right.grad(), EPS);
    }

    @Test
    void backwardSub() {
        Tensor left = Tensor.of(new float[] {5f, 6f}, 2).withGrad(true);
        Tensor right = Tensor.of(new float[] {1f, 2f}, 2).withGrad(true);
        left.sub(right).sum().backward();
        assertAllClose(new float[] {1f, 1f}, left.grad(), EPS);
        assertAllClose(new float[] {-1f, -1f}, right.grad(), EPS);
    }

    @Test
    void backwardMul() {
        Tensor left = Tensor.of(new float[] {2f, 3f}, 2).withGrad(true);
        Tensor right = Tensor.of(new float[] {4f, 5f}, 2).withGrad(true);
        left.mul(right).sum().backward();
        assertAllClose(new float[] {4f, 5f}, left.grad(), EPS);
        assertAllClose(new float[] {2f, 3f}, right.grad(), EPS);
    }

    @Test
    void backwardDiv() {
        Tensor left = Tensor.of(new float[] {6f, 8f}, 2).withGrad(true);
        Tensor right = Tensor.of(new float[] {2f, 4f}, 2).withGrad(true);
        left.div(right).sum().backward();
        assertAllClose(new float[] {0.5f, 0.25f}, left.grad(), EPS);
        assertAllClose(new float[] {-1.5f, -0.5f}, right.grad(), EPS);
    }

    @Test
    void backwardNeg() {
        Tensor input = Tensor.of(new float[] {1f, -2f}, 2).withGrad(true);
        input.neg().sum().backward();
        assertAllClose(new float[] {-1f, -1f}, input.grad(), EPS);
    }

    @Test
    void backwardMatmul() {
        Tensor left = matrix2d(new float[][] {{1f, 2f}}).withGrad(true);
        Tensor right = matrix2d(new float[][] {{3f}, {4f}}).withGrad(true);
        left.matmul(right).sum().backward();
        assertGradPresent(left);
        assertGradPresent(right);
        assertNoNaN(left.grad());
        assertNoNaN(right.grad());
    }

    @Test
    void backwardRelu() {
        Tensor input = Tensor.of(new float[] {-1f, 2f, 3f}, 3).withGrad(true);
        input.relu().sum().backward();
        assertAllClose(new float[] {0f, 1f, 1f}, input.grad(), EPS);
    }

    @Test
    void backwardSum() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f}, 3).withGrad(true);
        input.sum().backward();
        assertAllClose(new float[] {1f, 1f, 1f}, input.grad(), EPS);
    }

    @Test
    void backwardMean() {
        Tensor input = Tensor.of(new float[] {2f, 4f, 6f}, 3).withGrad(true);
        input.mean().backward();
        assertAllClose(new float[] {1f / 3f, 1f / 3f, 1f / 3f}, input.grad(), EPS);
    }

    @Test
    void backwardPow2() {
        Tensor input = Tensor.of(new float[] {2f, 3f}, 2).withGrad(true);
        input.pow2().sum().backward();
        assertAllClose(new float[] {4f, 6f}, input.grad(), EPS);
    }

    @Test
    void backwardReshape() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2).withGrad(true);
        input.reshape(4).sum().backward();
        assertAllClose(new float[] {1f, 1f, 1f, 1f}, input.grad(), EPS);
    }

    @Test
    void backwardExpand() {
        Tensor input = Tensor.of(new float[] {2f, 3f}, 2).withGrad(true);
        input.expand(2, 2).sum().backward();
        assertAllClose(new float[] {2f, 2f}, input.grad(), EPS);
    }

    @Test
    void backwardPermute() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2).withGrad(true);
        input.permute(1, 0).sum().backward();
        assertAllClose(new float[] {1f, 1f, 1f, 1f}, input.grad(), EPS);
    }

    @Test
    void backwardBiasAdd() {
        Tensor activations = matrix2d(new float[][] {{1f, 2f}, {3f, 4f}}).withGrad(true);
        Tensor bias = Tensor.of(new float[] {0.5f, -0.5f}, 2).withGrad(true);
        activations.add(bias.expand(2, 2)).sum().backward();
        assertGradPresent(activations);
        assertGradPresent(bias);
        assertAllClose(new float[] {2f, 2f}, bias.grad(), EPS);
    }

    @Test
    void gradcheckAddSubMulDivNeg() {
        Tensor addLeft = Tensor.of(new float[] {0.5f, -1f}, 2).withGrad(true);
        Tensor addRight = Tensor.of(new float[] {1f, 2f}, 2);
        GradCheck.assertGradClose(addLeft, new BinaryLossGraph(addLeft, addRight, OpKind.ADD));

        Tensor subLeft = Tensor.of(new float[] {2f, 3f}, 2).withGrad(true);
        Tensor subRight = Tensor.of(new float[] {1f, 1f}, 2);
        GradCheck.assertGradClose(subLeft, new BinaryLossGraph(subLeft, subRight, OpKind.SUB));

        Tensor mulLeft = Tensor.of(new float[] {0.5f, -1f}, 2).withGrad(true);
        Tensor mulRight = Tensor.of(new float[] {2f, 3f}, 2);
        GradCheck.assertGradClose(mulLeft, new BinaryLossGraph(mulLeft, mulRight, OpKind.MUL));

        Tensor divLeft = Tensor.of(new float[] {1f, 4f}, 2).withGrad(true);
        Tensor divRight = Tensor.of(new float[] {2f, 2f}, 2);
        GradCheck.assertGradClose(divLeft, new BinaryLossGraph(divLeft, divRight, OpKind.DIV));

        Tensor negInput = Tensor.of(new float[] {1f, -2f}, 2).withGrad(true);
        GradCheck.assertGradClose(negInput, new UnaryLossGraph(OpKind.NEG));
    }

    @Test
    void gradcheckSumMeanRelu() {
        Tensor sumInput = Tensor.of(new float[] {1f, 2f, 3f}, 3).withGrad(true);
        GradCheck.assertGradClose(sumInput, new UnaryLossGraph(OpKind.SUM));

        Tensor meanInput = Tensor.of(new float[] {1f, 2f, 3f}, 3).withGrad(true);
        GradCheck.assertGradClose(meanInput, new UnaryLossGraph(OpKind.MEAN));

        Tensor reluInput = Tensor.of(new float[] {-0.5f, 1f, 2f}, 3).withGrad(true);
        GradCheck.assertGradClose(reluInput, new UnaryLossGraph(OpKind.RELU));
    }

    @Test
    void fullGraphBackwardNoNaN() {
        Tensor input = matrix2d(new float[][] {{0.5f, -1f, 0.25f}}).withGrad(true);
        Tensor weight = matrix2d(new float[][] {{1f}, {2f}, {-1f}}).withGrad(true);
        Tensor bias = Tensor.of(new float[] {0.1f}, 1).withGrad(true);
        Tensor hidden = input.matmul(weight).add(bias.expand(1, 1));
        Tensor loss = hidden.relu().pow2().mean();
        loss.backward();
        assertNotNull(input.grad());
        assertNotNull(weight.grad());
        assertNotNull(bias.grad());
        assertNoNaN(input.grad());
        assertNoNaN(weight.grad());
        assertNoNaN(bias.grad());
    }

    private enum OpKind {
        ADD, SUB, MUL, DIV, NEG, SUM, MEAN, RELU
    }

    private static final class BinaryLossGraph implements GradCheck.ScalarLossGraph {
        private final Tensor right;
        private final OpKind kind;

        BinaryLossGraph(Tensor left, Tensor right, OpKind kind) {
            this.right = right;
            this.kind = kind;
        }

        @Override
        public Tensor loss(Tensor parameter) {
            Tensor combined;
            if (kind == OpKind.ADD) {
                combined = parameter.add(right);
            } else if (kind == OpKind.SUB) {
                combined = parameter.sub(right);
            } else if (kind == OpKind.MUL) {
                combined = parameter.mul(right);
            } else if (kind == OpKind.DIV) {
                combined = parameter.div(right);
            } else {
                throw new IllegalStateException("unsupported binary op");
            }
            return combined.mul(combined).sum();
        }
    }

    private static final class UnaryLossGraph implements GradCheck.ScalarLossGraph {
        private final OpKind kind;

        UnaryLossGraph(OpKind kind) {
            this.kind = kind;
        }

        @Override
        public Tensor loss(Tensor parameter) {
            Tensor value = parameter;
            if (kind == OpKind.NEG) {
                value = parameter.neg();
            } else if (kind == OpKind.RELU) {
                value = parameter.relu();
            } else if (kind == OpKind.SUM) {
                return parameter.sum();
            } else if (kind == OpKind.MEAN) {
                return parameter.mean();
            }
            return value.mul(value).sum();
        }
    }
}
