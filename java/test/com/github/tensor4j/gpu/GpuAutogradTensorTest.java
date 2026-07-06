/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import com.github.tensor4j.gpu.core.GpuAutogradTensor;
import com.github.tensor4j.gpu.ref.CpuDevice;

/**
 * Tests for GPU autograd (computation graph + backward).
 */
class GpuAutogradTensorTest {

    @Test
    void addForward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{4f, 5f, 6f}, 3);
            GpuAutogradTensor c = a.add(b);
            assertArrayEquals(new float[]{5f, 7f, 9f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void addBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3).requiresGrad(true);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{4f, 5f, 6f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.add(b);
            c.backward();
            assertArrayEquals(new float[]{1f, 1f, 1f}, a.grad().toHost(), 1e-5f);
            assertArrayEquals(new float[]{1f, 1f, 1f}, b.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void subBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{10f, 20f, 30f}, 3).requiresGrad(true);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.sub(b);
            c.backward();
            assertArrayEquals(new float[]{1f, 1f, 1f}, a.grad().toHost(), 1e-5f);
            assertArrayEquals(new float[]{-1f, -1f, -1f}, b.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void mulBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{2f, 3f, 4f}, 3).requiresGrad(true);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{5f, 6f, 7f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.mul(b);
            c.backward();
            assertArrayEquals(new float[]{5f, 6f, 7f}, a.grad().toHost(), 1e-5f);
            assertArrayEquals(new float[]{2f, 3f, 4f}, b.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void divBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{10f, 20f, 30f}, 3).requiresGrad(true);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{2f, 4f, 5f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.div(b);
            c.backward();
            assertArrayEquals(new float[]{0.5f, 0.25f, 0.2f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void reluBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{-2f, -1f, 0f, 1f, 2f}, 5).requiresGrad(true);
            GpuAutogradTensor c = a.relu();
            c.backward();
            assertArrayEquals(new float[]{0f, 0f, 0f, 1f, 1f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void negBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, -2f, 0f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.neg();
            c.backward();
            assertArrayEquals(new float[]{-1f, -1f, -1f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void scaleBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.scale(2.5f);
            c.backward();
            assertArrayEquals(new float[]{2.5f, 2.5f, 2.5f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void expBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{0f, 1f, 2f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.exp();
            c.backward();
            float[] expected = {(float) Math.exp(0), (float) Math.exp(1), (float) Math.exp(2)};
            assertArrayEquals(expected, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void logBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, (float) Math.E}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.log();
            c.backward();
            float[] expected = {1f, 0.5f, 1f / (float) Math.E};
            assertArrayEquals(expected, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void sqrtBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 4f, 9f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.sqrt();
            c.backward();
            float[] expected = {0.5f, 0.25f, 1f / 6f};
            assertArrayEquals(expected, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void sigmoidBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{0f, 1f, 2f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.sigmoid();
            c.backward();
            float[] s = {0.5f, 0.7310586f, 0.880797f};
            float[] expected = {s[0] * (1 - s[0]), s[1] * (1 - s[1]), s[2] * (1 - s[2])};
            assertArrayEquals(expected, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void tanhBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{0f, 1f, 2f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.tanh();
            c.backward();
            float[] t = {0f, 0.761594f, 0.964028f};
            float[] expected = {1 - t[0] * t[0], 1 - t[1] * t[1], 1 - t[2] * t[2]};
            assertArrayEquals(expected, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void matmulBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 2, 2).requiresGrad(true);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{5f, 6f, 7f, 8f}, 2, 2).requiresGrad(true);
            GpuAutogradTensor c = a.matmul(b);
            c.backward();
            assertArrayEquals(new float[]{11f, 15f, 11f, 15f}, a.grad().toHost(), 1e-5f);
            assertArrayEquals(new float[]{4f, 4f, 6f, 6f}, b.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void squaredDifferenceBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3).requiresGrad(true);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{2f, 2f, 2f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.squaredDifference(b);
            c.backward();
            assertArrayEquals(new float[]{-2f, 0f, 2f}, a.grad().toHost(), 1e-5f);
            assertArrayEquals(new float[]{2f, 0f, -2f}, b.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void sumBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f}, 5).requiresGrad(true);
            GpuAutogradTensor c = a.sum();
            c.backward();
            assertArrayEquals(new float[]{1f, 1f, 1f, 1f, 1f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void chainMulAddBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{2f, 3f}, 2).requiresGrad(true);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{4f, 5f}, 2).requiresGrad(true);
            GpuAutogradTensor c = a.mul(b);
            GpuAutogradTensor d = c.add(a);
            d.backward();
            assertArrayEquals(new float[]{5f, 6f}, a.grad().toHost(), 1e-5f);
            assertArrayEquals(new float[]{2f, 3f}, b.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void noGradRequiredDoesNotAccumulate() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f}, 2).requiresGrad(true);
            GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{3f, 4f}, 2);
            GpuAutogradTensor c = a.add(b);
            c.backward();
            assertArrayEquals(new float[]{1f, 1f}, a.grad().toHost(), 1e-5f);
            assertNull(b.grad());
        }
    }

    @Test
    void backwardMultiplePaths() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{2f}, 1).requiresGrad(true);
            GpuAutogradTensor b = a.mul(a);
            GpuAutogradTensor c = b.add(a);
            c.backward();
            assertArrayEquals(new float[]{5f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void powBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{2f, 3f}, 2).requiresGrad(true);
            GpuAutogradTensor e = GpuAutogradTensor.fromHost(dev, new float[]{3f, 2f}, 2).requiresGrad(true);
            GpuAutogradTensor c = a.pow(e);
            c.backward();
            // f(x,y) = x^y  =>  df/dx = y*x^(y-1), df/dy = x^y*ln(x)
            assertArrayEquals(new float[]{3f * 4f, 2f * 3f}, a.grad().toHost(), 1e-4f);
            assertArrayEquals(new float[]{8f * (float) Math.log(2), 9f * (float) Math.log(3)}, e.grad().toHost(), 1e-4f);
        }
    }

    @Test
    void meanBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 4).requiresGrad(true);
            GpuAutogradTensor c = a.mean();
            c.backward();
            assertArrayEquals(new float[]{0.25f, 0.25f, 0.25f, 0.25f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void absBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{-2f, -1f, 0f, 1f, 2f}, 5).requiresGrad(true);
            GpuAutogradTensor c = a.abs();
            c.backward();
            assertArrayEquals(new float[]{-1f, -1f, 0f, 1f, 1f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void clipBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{-5f, -1f, 0f, 2f, 10f}, 5).requiresGrad(true);
            GpuAutogradTensor c = a.clip(-2f, 4f);
            c.backward();
            assertArrayEquals(new float[]{0f, 1f, 1f, 1f, 0f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void transposeBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 2, 2).requiresGrad(true);
            GpuAutogradTensor c = a.transpose();
            c.backward();
            assertArrayEquals(new float[]{1f, 1f, 1f, 1f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void sinBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{0f, (float) (Math.PI / 2)}, 2).requiresGrad(true);
            GpuAutogradTensor c = a.sin();
            c.backward();
            // d(sin)/dx = cos(x)
            assertArrayEquals(new float[]{1f, 0f}, a.grad().toHost(), 1e-4f);
        }
    }

    @Test
    void cosBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{0f, (float) (Math.PI / 2)}, 2).requiresGrad(true);
            GpuAutogradTensor c = a.cos();
            c.backward();
            assertArrayEquals(new float[]{0f, -1f}, a.grad().toHost(), 1e-4f);
        }
    }

    @Test
    void reshapeBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 2, 2).requiresGrad(true);
            GpuAutogradTensor r = a.reshape(4);
            GpuAutogradTensor s = r.sum();
            s.backward();
            assertArrayEquals(new float[]{1f, 1f, 1f, 1f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void squeezeBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 1, 3).requiresGrad(true);
            GpuAutogradTensor s = a.squeeze();
            GpuAutogradTensor sum = s.sum();
            sum.backward();
            assertArrayEquals(new float[]{1f, 1f, 1f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void unsqueezeBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3).requiresGrad(true);
            GpuAutogradTensor u = a.unsqueeze(0);
            GpuAutogradTensor sum = u.sum();
            sum.backward();
            assertArrayEquals(new float[]{1f, 1f, 1f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void geluBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{-1f, 0f, 1f}, 3).requiresGrad(true);
            GpuAutogradTensor c = a.gelu();
            c.backward();
            // GELU is roughly identity near 0, gradient should be ~0.5 at x=0
            assertNotNull(a.grad());
            assertEquals(0.5f, a.grad().toHost()[1], 0.1f);
        }
    }

    @Test
    void leakyReluBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{-2f, -1f, 0f, 1f, 2f}, 5).requiresGrad(true);
            GpuAutogradTensor c = a.leakyRelu(0.1f);
            c.backward();
            float[] g = a.grad().toHost();
            assertEquals(0.1f, g[0], 1e-5f);
            assertEquals(0.1f, g[1], 1e-5f);
            assertEquals(0.1f, g[2], 1e-5f); // x=0, > 0 is false so alpha gradient
            assertEquals(1f, g[3], 1e-5f);
            assertEquals(1f, g[4], 1e-5f);
        }
    }

    @Test
    void expandBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f}, 1, 2).requiresGrad(true);
            GpuAutogradTensor e = a.expand(3, 2);
            GpuAutogradTensor sum = e.sum();
            sum.backward();
            assertArrayEquals(new float[]{3f, 3f}, a.grad().toHost(), 1e-5f);
        }
    }

    @Test
    void chainReshapeGeluBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{-1f, 0f, 1f, 2f}, 2, 2).requiresGrad(true);
            GpuAutogradTensor r = a.reshape(4);
            GpuAutogradTensor g = r.gelu();
            GpuAutogradTensor s = g.sum();
            s.backward();
            assertNotNull(a.grad());
            assertEquals(4, a.grad().numel());
        }
    }

    @Test
    void dropoutBackwardPreservesGradient() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 4).requiresGrad(true);
            GpuAutogradTensor d = a.dropout(0.5f, new java.util.Random(42));
            GpuAutogradTensor s = d.sum();
            s.backward();
            assertNotNull(a.grad());
            assertEquals(4, a.grad().numel());
        }
    }

    @Test
    void binaryCrossEntropyBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor pred = GpuAutogradTensor.fromHost(dev, new float[]{0.7f, 0.3f}, 2).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev, new float[]{1f, 0f}, 2);
            GpuAutogradTensor loss = pred.binaryCrossEntropy(target);
            loss.backward();
            assertNotNull(pred.grad());
            assertEquals(2, pred.grad().numel());
        }
    }

    @Test
    void crossEntropyBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor logits = GpuAutogradTensor.fromHost(dev,
                new float[]{2f, 1f, 0.1f, 0f, 3f, 1f}, 2, 3).requiresGrad(true);
            GpuAutogradTensor targets = GpuAutogradTensor.fromHost(dev,
                new float[]{0f, 1f}, 2);
            GpuAutogradTensor loss = logits.crossEntropy(targets);
            loss.backward();
            assertNotNull(logits.grad());
            assertArrayEquals(new int[]{2, 3}, logits.grad().shape());
            // Gradient should be softmax - one_hot
            float[] g = logits.grad().toHost();
            // Row 0: softmax([2,1,0.1]) - [1,0,0] ≈ [-0.34, 0.24, 0.10]
            assertTrue(g[0] < 0f, "g[0] should be negative: " + g[0]);
            assertTrue(g[1] > 0f, "g[1] should be positive: " + g[1]);
        }
    }

    @Test
    void crossEntropyThroughMatmulBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor x = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 2f, 3f, 4f}, 2, 2);
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev,
                new float[]{0.1f, 0.2f, 0.3f, 0.4f}, 2, 2).requiresGrad(true);
            GpuAutogradTensor targets = GpuAutogradTensor.fromHost(dev,
                new float[]{0f, 1f}, 2);
            GpuAutogradTensor logits = x.matmul(w);
            GpuAutogradTensor loss = logits.crossEntropy(targets);
            loss.backward();
            assertNotNull(w.grad(), "weight gradient should not be null");
            assertEquals(4, w.grad().numel());
        }
    }

    @Test
    void layerNormBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor x = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3).requiresGrad(true);
            GpuAutogradTensor gamma = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 1f, 1f}, 3).requiresGrad(true);
            GpuAutogradTensor beta = GpuAutogradTensor.fromHost(dev,
                new float[]{0f, 0f, 0f}, 3).requiresGrad(true);
            GpuAutogradTensor ln = x.layerNorm(gamma, beta, 1e-5f);
            GpuAutogradTensor s = ln.sum();
            s.backward();
            assertNotNull(x.grad());
            assertNotNull(gamma.grad());
            assertNotNull(beta.grad());
        }
    }

    @Test
    void mseLossBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor pred = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 2f, 3f}, 3).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 0f, 3f}, 3);
            GpuAutogradTensor loss = pred.mseLoss(target);
            loss.backward();
            assertNotNull(pred.grad());
            // d/dpred (sum((pred-target)^2)) = 2*(pred-target)
            float[] g = pred.grad().toHost();
            assertEquals(0f, g[0], 1e-5f);
            assertEquals(4f, g[1], 1e-5f);
            assertEquals(0f, g[2], 1e-5f);
        }
    }

    @Test
    void embeddingBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor weight = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 3, 2).requiresGrad(true);
            GpuAutogradTensor indices = GpuAutogradTensor.fromHost(dev,
                new float[]{0f, 2f, 0f}, 3);
            GpuAutogradTensor emb = weight.embedding(indices);
            GpuAutogradTensor s = emb.sum();
            s.backward();
            assertNotNull(weight.grad());
            // Index 0 appears twice, so grad should be 2 for those entries
            float[] g = weight.grad().toHost();
            assertEquals(2f, g[0], 1e-5f);
            assertEquals(2f, g[1], 1e-5f);
            assertEquals(0f, g[2], 1e-5f);
            assertEquals(0f, g[3], 1e-5f);
            assertEquals(1f, g[4], 1e-5f);
            assertEquals(1f, g[5], 1e-5f);
        }
    }

    @Test
    void conv2dBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor input = GpuAutogradTensor.fromHost(dev,
                new float[]{1,2,3,4,5,6,7,8,9}, 1, 1, 3, 3).requiresGrad(true);
            GpuAutogradTensor kernel = GpuAutogradTensor.fromHost(dev,
                new float[]{1,0,0,1}, 1, 1, 2, 2).requiresGrad(true);
            GpuAutogradTensor conv = input.conv2d(kernel, 1, 0);
            GpuAutogradTensor s = conv.sum();
            s.backward();
            assertNotNull(input.grad());
            assertNotNull(kernel.grad());
            assertEquals(4, kernel.grad().numel());
        }
    }

    @Test
    void maxPool2dBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor input = GpuAutogradTensor.fromHost(dev,
                new float[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, 1, 1, 4, 4).requiresGrad(true);
            GpuAutogradTensor pool = input.maxPool2d(2, 2, 0);
            GpuAutogradTensor s = pool.sum();
            s.backward();
            assertNotNull(input.grad());
            assertEquals(16, input.grad().numel());
            // Each max element should get gradient 1
            float[] g = input.grad().toHost();
            assertEquals(1f, g[5], 1e-5f); // max of [1,2,5,6] is 6 at index 5
            assertEquals(1f, g[7], 1e-5f); // max of [3,4,7,8] is 8 at index 7
            assertEquals(1f, g[13], 1e-5f); // max of [9,10,13,14] is 14 at index 13
            assertEquals(1f, g[15], 1e-5f); // max of [11,12,15,16] is 16 at index 15
            assertEquals(0f, g[0], 1e-5f); // non-max elements get 0
        }
    }

    @Test
    void softmax2dBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor x = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3).requiresGrad(true);
            GpuAutogradTensor sm = x.softmax();
            GpuAutogradTensor s = sm.sum();
            s.backward();
            assertNotNull(x.grad());
            // Sum of softmax is constant (1 per row), so gradient of sum w.r.t. x should be 0
            float[] g = x.grad().toHost();
            for (float v : g) {
                assertEquals(0f, v, 1e-5f);
            }
        }
    }

    @Test
    void batchNorm1dBackward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor x = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 10f, 2f, 20f, 3f, 30f}, 3, 2).requiresGrad(true);
            GpuAutogradTensor gamma = GpuAutogradTensor.fromHost(dev,
                new float[]{1f, 1f}, 2).requiresGrad(true);
            GpuAutogradTensor beta = GpuAutogradTensor.fromHost(dev,
                new float[]{0f, 0f}, 2).requiresGrad(true);
            GpuAutogradTensor bn = x.batchNorm1d(gamma, beta, 1e-5f);
            GpuAutogradTensor s = bn.sum();
            s.backward();
            assertNotNull(x.grad());
            assertNotNull(gamma.grad());
            assertNotNull(beta.grad());
        }
    }
}
