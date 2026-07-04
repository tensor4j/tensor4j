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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.MaxPool2dResult;
import com.github.tensor4j.core.Pool2dArg;
import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class LazyConvPoolExtendedTest {

    private static final float EPS = 1e-3f;

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void batchNorm2dLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f};
        Tensor x = Tensor.of(xData, 1, 2, 1, 2);
        Tensor weight = Tensor.of(new float[] {1f, 2f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 1f}, 2);
        Tensor mean = Tensor.of(new float[] {0f, 1f}, 2);
        Tensor var = Tensor.of(new float[] {1f, 1f}, 2);
        Tensor eager = x.batchNorm2d(weight, bias, mean, var, 1e-5f);
        LazyTensor lazy = LazyTensor.of(xData, 1, 2, 1, 2).batchNorm2d(
                LazyTensor.of(new float[] {1f, 2f}, 2), LazyTensor.of(new float[] {0f, 1f}, 2),
                LazyTensor.of(new float[] {0f, 1f}, 2), LazyTensor.of(new float[] {1f, 1f}, 2), 1e-5f);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void batchNorm2dLazyBackwardMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor weight = Tensor.of(new float[] {1f, 2f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 1f}, 2);
        Tensor mean = Tensor.of(new float[] {0f, 1f}, 2);
        Tensor var = Tensor.of(new float[] {1f, 1f}, 2);
        Tensor eager = Tensor.of(data, 1, 2, 1, 2).withGrad(true);
        eager.batchNorm2d(weight, bias, mean, var, 1e-5f).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 1, 2, 1, 2).withGrad(true);
        lazy.batchNorm2d(LazyTensor.of(new float[] {1f, 2f}, 2), LazyTensor.of(new float[] {0f, 1f}, 2),
                LazyTensor.of(new float[] {0f, 1f}, 2), LazyTensor.of(new float[] {1f, 1f}, 2), 1e-5f)
                .sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void batchNorm2dTrainLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        Tensor weight = Tensor.of(new float[] {1f, 1f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f}, 2);
        Tensor eager = Tensor.of(xData, 1, 2, 2, 2).batchNorm2dTrain(weight, bias, 1e-5f);
        LazyTensor lazy = LazyTensor.of(xData, 1, 2, 2, 2).batchNorm2dTrain(
                LazyTensor.of(new float[] {1f, 1f}, 2), LazyTensor.of(new float[] {0f, 0f}, 2), 1e-5f);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void batchNorm2dTrainLazyBackwardMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] weightData = new float[] {1f, 1f};
        float[] biasData = new float[] {0f, 0f};
        Tensor eagerX = Tensor.of(data, 1, 2, 2, 2).withGrad(true);
        Tensor eagerW = Tensor.of(weightData, 2).withGrad(true);
        Tensor eagerB = Tensor.of(biasData, 2).withGrad(true);
        eagerX.batchNorm2dTrain(eagerW, eagerB, 1e-5f).sum().backward();
        LazyTensor lazyX = LazyTensor.of(data, 1, 2, 2, 2).withGrad(true);
        LazyTensor lazyW = LazyTensor.of(weightData, 2).withGrad(true);
        LazyTensor lazyB = LazyTensor.of(biasData, 2).withGrad(true);
        lazyX.batchNorm2dTrain(lazyW, lazyB, 1e-5f).sum().backward();
        assertAllClose(eagerX.grad(), lazyX.grad(), EPS);
        assertAllClose(eagerW.grad(), lazyW.grad(), EPS);
        assertAllClose(eagerB.grad(), lazyB.grad(), EPS);
    }

    @Test
    void adaptiveAvgPoolLazyMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f};
        Tensor eager = Tensor.of(data, 1, 1, 4, 4).adaptiveAvgPool2d(2, 2);
        LazyTensor lazy = LazyTensor.of(data, 1, 1, 4, 4).adaptiveAvgPool2d(2, 2);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void maxUnpoolLazyMatchesEager() {
        float[] data = new float[16];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1f;
        }
        Tensor t = Tensor.of(data, 1, 1, 4, 4);
        MaxPool2dResult pooled = t.maxPool2dWithIndices(2, 2);
        Tensor eager = pooled.output.maxUnpool2d(pooled.indices, Pool2dArg.maxPacked(2, 2), t.shape().dims());
        LazyTensor lazy = LazyTensor.of(pooled.output.toFlatArray(), 1, 1, 2, 2).maxUnpool2d(
                LazyTensor.of(pooled.indices.toFlatArray(), 1, 1, 2, 2), Pool2dArg.maxPacked(2, 2), t.shape().dims());
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void maxUnpoolLazyBackwardMatchesEager() {
        float[] data = new float[16];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1f;
        }
        Tensor t = Tensor.of(data, 1, 1, 4, 4);
        MaxPool2dResult pooled = t.maxPool2dWithIndices(2, 2);
        Tensor eagerVals = pooled.output.withGrad(true);
        eagerVals.maxUnpool2d(pooled.indices, Pool2dArg.maxPacked(2, 2), t.shape().dims()).sum().backward();
        LazyTensor lazyVals = LazyTensor.of(pooled.output.toFlatArray(), 1, 1, 2, 2).withGrad(true);
        lazyVals.maxUnpool2d(LazyTensor.of(pooled.indices.toFlatArray(), 1, 1, 2, 2), Pool2dArg.maxPacked(2, 2),
                t.shape().dims())
                .sum().backward();
        assertAllClose(eagerVals.grad(), lazyVals.grad(), EPS);
    }

    @Test
    void maxUnpoolBuildsUOp() {
        LazyTensor values = LazyTensor.of(new float[] {1f, 2f}, 1, 1, 1, 2);
        LazyTensor indices = LazyTensor.of(new float[] {0f, 3f}, 1, 1, 1, 2);
        assertTrue(hasKind(values.maxUnpool2d(indices, Pool2dArg.maxPacked(2, 2), new int[] {1, 1, 2, 2}).uop(),
                LazyUOp.Kind.MAX_UNPOOL2D));
    }

    @Test
    void layerNormGenericLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor weight = Tensor.of(new float[] {1f, 1f, 1f}, 3);
        Tensor bias = Tensor.of(new float[] {0f, 0f, 0f}, 3);
        Tensor eager = Tensor.of(xData, 2, 3).layerNorm(weight, bias, new int[] {3}, 1e-5f);
        LazyTensor lazy = LazyTensor.of(xData, 2, 3).layerNorm(
                LazyTensor.of(new float[] {1f, 1f, 1f}, 3), LazyTensor.of(new float[] {0f, 0f, 0f}, 3),
                new int[] {3}, 1e-5f);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void layerNorm2dLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f};
        Tensor weight = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 2, 1, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f, 0f, 0f}, 2, 1, 2);
        Tensor eager = Tensor.of(xData, 1, 2, 1, 2).layerNorm2d(weight, bias, 1e-5f);
        LazyTensor lazy = LazyTensor.of(xData, 1, 2, 1, 2).layerNorm2d(
                LazyTensor.of(new float[] {1f, 1f, 1f, 1f}, 2, 1, 2),
                LazyTensor.of(new float[] {0f, 0f, 0f, 0f}, 2, 1, 2), 1e-5f);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void layerNorm2dLazyBackwardMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor weight = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 2, 1, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f, 0f, 0f}, 2, 1, 2);
        Tensor eager = Tensor.of(data, 1, 2, 1, 2).withGrad(true);
        eager.layerNorm2d(weight, bias, 1e-5f).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 1, 2, 1, 2).withGrad(true);
        lazy.layerNorm2d(LazyTensor.of(new float[] {1f, 1f, 1f, 1f}, 2, 1, 2),
                LazyTensor.of(new float[] {0f, 0f, 0f, 0f}, 2, 1, 2), 1e-5f)
                .sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void groupNorm2dLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f};
        Tensor weight = Tensor.of(new float[] {1f, 1f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f}, 2);
        Tensor eager = Tensor.of(xData, 1, 2, 1, 2).groupNorm2d(2, weight, bias, 1e-5f);
        LazyTensor lazy = LazyTensor.of(xData, 1, 2, 1, 2).groupNorm2d(2,
                LazyTensor.of(new float[] {1f, 1f}, 2), LazyTensor.of(new float[] {0f, 0f}, 2), 1e-5f);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void groupNorm2dLazyBackwardMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor weight = Tensor.of(new float[] {1f, 1f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f}, 2);
        Tensor eager = Tensor.of(data, 1, 2, 1, 2).withGrad(true);
        eager.groupNorm2d(2, weight, bias, 1e-5f).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 1, 2, 1, 2).withGrad(true);
        lazy.groupNorm2d(2, LazyTensor.of(new float[] {1f, 1f}, 2), LazyTensor.of(new float[] {0f, 0f}, 2), 1e-5f)
                .sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void dropoutLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f};
        Tensor eager = Tensor.of(xData, 1, 2, 1, 2).dropout(0.5f, 42L);
        LazyTensor lazy = LazyTensor.of(xData, 1, 2, 1, 2).dropout(0.5f, 42L);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void dropoutLazyBackwardMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor eager = Tensor.of(data, 1, 2, 1, 2).withGrad(true);
        eager.dropout(0.5f, 42L).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 1, 2, 1, 2).withGrad(true);
        lazy.dropout(0.5f, 42L).sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    private static boolean hasKind(LazyUOp root, LazyUOp.Kind kind) {
        for (LazyUOp node : LazyGraph.toposort(root)) {
            if (node.op() == kind) {
                return true;
            }
        }
        return false;
    }
}
