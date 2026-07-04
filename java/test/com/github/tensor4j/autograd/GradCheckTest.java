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

import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.support.GradCheck;
import org.junit.jupiter.api.Test;

/** Layer 2b: finite-difference gradcheck on flat float parameters. */
class GradCheckTest {

    @Test
    void matmulGradMatchesFiniteDifference() {
        Tensor left = Tensor.of(new float[] {0.5f, -1f, 2f, 0.25f}, 2, 2).withGrad(true);
        Tensor right = Tensor.of(new float[] {1f, 2f, -0.5f, 0.75f}, 2, 2);
        GradCheck.assertGradClose(left, new MatmulLossGraph(right));
    }

    @Test
    void addGradMatchesFiniteDifference() {
        Tensor left = Tensor.of(new float[] {1f, -2f, 0.5f}, 3).withGrad(true);
        Tensor right = Tensor.of(new float[] {0.25f, 1f, -0.5f}, 3);
        GradCheck.assertGradClose(left, new AddLossGraph(right));
    }

    @Test
    void reluGradMatchesFiniteDifference() {
        Tensor input = Tensor.of(new float[] {-1f, 0.5f, 2f}, 3).withGrad(true);
        GradCheck.assertGradClose(input, new ReluLossGraph());
    }

    @Test
    void expandAddGradMatchesFiniteDifference() {
        Tensor bias = Tensor.of(new float[] {0.1f, -0.2f}, 1, 2).withGrad(true);
        Tensor batch = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2);
        GradCheck.assertGradClose(bias, new ExpandAddLossGraph(batch));
    }

    private static final class MatmulLossGraph implements GradCheck.ScalarLossGraph {
        private final Tensor right;

        MatmulLossGraph(Tensor right) {
            this.right = right;
        }

        @Override
        public Tensor loss(Tensor left) {
            return left.matmul(right).mul(left.matmul(right)).sum();
        }
    }

    private static final class AddLossGraph implements GradCheck.ScalarLossGraph {
        private final Tensor right;

        AddLossGraph(Tensor right) {
            this.right = right;
        }

        @Override
        public Tensor loss(Tensor left) {
            Tensor sum = left.add(right);
            return sum.mul(sum).sum();
        }
    }

    private static final class ReluLossGraph implements GradCheck.ScalarLossGraph {
        @Override
        public Tensor loss(Tensor input) {
            Tensor activated = input.relu();
            return activated.mul(activated).sum();
        }
    }

    private static final class ExpandAddLossGraph implements GradCheck.ScalarLossGraph {
        private final Tensor batch;

        ExpandAddLossGraph(Tensor batch) {
            this.batch = batch;
        }

        @Override
        public Tensor loss(Tensor bias) {
            Tensor out = batch.add(bias.expand(batch.shape().dim(0), batch.shape().dim(1)));
            return out.mul(out).sum();
        }
    }
}
