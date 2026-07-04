/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core;

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.support.GradCheck;
import org.junit.jupiter.api.Test;

/** Batch norm, adaptive pool, max_unpool, layer/group norm, dropout (tinygrad doc examples). */
class ConvPoolExtendedOpsTest {

    private static final float EPS = 1e-3f;

    @Test
    void batchNorm2dEval() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 1, 2);
        Tensor weight = Tensor.of(new float[] {1f, 2f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 1f}, 2);
        Tensor mean = Tensor.of(new float[] {0f, 1f}, 2);
        Tensor var = Tensor.of(new float[] {1f, 1f}, 2);
        assertAllClose(new float[] {1f, 2f, 5f, 7f}, x.batchNorm2d(weight, bias, mean, var, 1e-5f), EPS);
    }

    @Test
    void batchNorm2dTrainForward() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 1, 2);
        Tensor weight = Tensor.of(new float[] {1f, 1f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f}, 2);
        Tensor out = x.batchNorm2dTrain(weight, bias, 1e-5f);
        assertEquals(4, out.numel());
    }

    @Test
    void batchNorm2dTrainGradcheck() {
        float[] xData = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        Tensor weight = Tensor.of(new float[] {1f, 1f}, 2).withGrad(true);
        GradCheck.assertGradClose(weight, param -> {
            Tensor x = Tensor.of(xData, 1, 2, 2, 2);
            Tensor bias = Tensor.of(new float[] {0f, 0f}, 2);
            return x.batchNorm2dTrain(param, bias, 1e-5f).sum();
        });
    }

    @Test
    void adaptiveAvgPool2d() {
        float[] data = new float[16];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1f;
        }
        Tensor x = Tensor.of(data, 1, 1, 4, 4);
        assertAllClose(new float[] {7.5f, 9.5f}, x.adaptiveAvgPool2d(1, 2), EPS);
    }

    @Test
    void adaptiveMaxPool2d() {
        float[] data = new float[16];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1f;
        }
        Tensor x = Tensor.of(data, 1, 1, 4, 4);
        assertAllClose(new float[] {14f, 16f}, x.adaptiveMaxPool2d(1, 2), EPS);
    }

    @Test
    void maxUnpool2dTinygradExample() {
        float[] data = new float[16];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1f;
        }
        Tensor t = Tensor.of(data, 1, 1, 4, 4);
        MaxPool2dResult pooled = t.maxPool2dWithIndices(2, 2);
        assertAllClose(new float[] {6f, 8f, 14f, 16f}, pooled.output, EPS);
        Tensor restored = pooled.output.maxUnpool2d(pooled.indices, Pool2dArg.maxPacked(2, 2), t.shape().dims());
        assertAllClose(new float[] {
                0f, 0f, 0f, 0f,
                0f, 6f, 0f, 8f,
                0f, 0f, 0f, 0f,
                0f, 14f, 0f, 16f
        }, restored, EPS);
    }

    @Test
    void maxUnpoolBackwardGradcheck() {
        float[] data = new float[16];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1f;
        }
        Tensor t = Tensor.of(data, 1, 1, 4, 4);
        MaxPool2dResult pooled = t.maxPool2dWithIndices(2, 2);
        Tensor values = pooled.output.withGrad(true);
        GradCheck.assertGradClose(values, param -> {
            Tensor restored = param.maxUnpool2d(pooled.indices, Pool2dArg.maxPacked(2, 2), t.shape().dims());
            return restored.sum();
        });
    }

    @Test
    void layerNormGenericForward() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
        Tensor weight = Tensor.of(new float[] {1f, 1f, 1f}, 3);
        Tensor bias = Tensor.of(new float[] {0f, 0f, 0f}, 3);
        Tensor out = x.layerNorm(weight, bias, new int[] {3}, 1e-5f);
        float invStd0 = 1f / (float) Math.sqrt(2f / 3f + 1e-5f);
        assertAllClose(new float[] {
                (1f - 2f) * invStd0, (2f - 2f) * invStd0, (3f - 2f) * invStd0,
                (4f - 5f) * invStd0, (5f - 5f) * invStd0, (6f - 5f) * invStd0
        }, out, EPS);
    }

    @Test
    void layerNorm2dForward() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 1, 2);
        Tensor weight = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 2, 1, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f, 0f, 0f}, 2, 1, 2);
        Tensor out = x.layerNorm2d(weight, bias, 1e-5f);
        float mean = 2.5f;
        float var = 1.25f;
        float invStd = 1f / (float) Math.sqrt(var + 1e-5f);
        assertAllClose(new float[] {
                (1f - mean) * invStd,
                (2f - mean) * invStd,
                (3f - mean) * invStd,
                (4f - mean) * invStd
        }, out, EPS);
    }

    @Test
    void layerNorm2dGradcheck() {
        float[] xData = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        Tensor weight = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 2, 2, 1).withGrad(true);
        GradCheck.assertGradClose(weight, param -> {
            Tensor x = Tensor.of(xData, 2, 2, 2, 1);
            Tensor bias = Tensor.of(new float[] {0f, 0f, 0f, 0f}, 2, 2, 1);
            return x.layerNorm2d(param, bias, 1e-5f).sum();
        });
    }

    @Test
    void groupNormGenericMatchesNchw() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 1, 2);
        Tensor weight = Tensor.of(new float[] {1f, 1f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f}, 2);
        assertAllClose(x.groupNorm2d(2, weight, bias, 1e-5f), x.groupNorm(2, weight, bias, 1, 1e-5f), EPS);
    }

    @Test
    void groupNorm2dForward() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 1, 2);
        Tensor weight = Tensor.of(new float[] {1f, 1f}, 2);
        Tensor bias = Tensor.of(new float[] {0f, 0f}, 2);
        Tensor out = x.groupNorm2d(2, weight, bias, 1e-5f);
        float mean0 = 1.5f;
        float var0 = 0.25f;
        float invStd0 = 1f / (float) Math.sqrt(var0 + 1e-5f);
        float mean1 = 3.5f;
        float var1 = 0.25f;
        float invStd1 = 1f / (float) Math.sqrt(var1 + 1e-5f);
        assertAllClose(new float[] {
                (1f - mean0) * invStd0,
                (2f - mean0) * invStd0,
                (3f - mean1) * invStd1,
                (4f - mean1) * invStd1
        }, out, EPS);
    }

    @Test
    void groupNorm2dGradcheck() {
        float[] xData = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        Tensor weight = Tensor.of(new float[] {1f, 1f}, 2).withGrad(true);
        GradCheck.assertGradClose(weight, param -> {
            Tensor x = Tensor.of(xData, 1, 2, 2, 2);
            Tensor bias = Tensor.of(new float[] {0f, 0f}, 2);
            return x.groupNorm2d(2, param, bias, 1e-5f).sum();
        });
    }

    @Test
    void dropoutForwardWithSeed() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 1, 2);
        Tensor mask = DropoutMath.sampleMask(new int[] {1, 2, 1, 2}, 0.5f, 42L);
        assertAllClose(x.dropoutWithMask(mask, 0.5f), x.dropout(0.5f, 42L), EPS);
    }

    @Test
    void dropoutWithMaskForward() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 1, 2);
        Tensor mask = Tensor.of(new float[] {1f, 0f, 1f, 1f}, 1, 2, 1, 2);
        assertAllClose(new float[] {2f, 0f, 6f, 8f}, x.dropoutWithMask(mask, 0.5f), EPS);
    }

    @Test
    void dropoutGradcheck() {
        float[] xData = new float[] {1f, 2f, 3f, 4f};
        Tensor x = Tensor.of(xData, 1, 2, 1, 2).withGrad(true);
        GradCheck.assertGradClose(x, param -> param.dropout(0.5f, 42L).sum());
    }
}
