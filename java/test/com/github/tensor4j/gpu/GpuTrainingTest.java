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
import com.github.tensor4j.gpu.core.*;
import com.github.tensor4j.gpu.ref.*;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * End-to-end training tests using GPU autograd tensor and optimizers.
 * Runs on CPU reference backend; no GPU hardware required.
 */
class GpuTrainingTest {

    @Test
    void mlpOverfitsSmallDataset() {
        try (CpuDevice dev = new CpuDevice()) {
            Random rng = new Random(42);
            int n = 32;
            float[] xData = new float[n * 2];
            float[] yData = new float[n];
            for (int i = 0; i < n; i++) {
                double angle = 2 * Math.PI * i / n;
                float r = (i < n / 2) ? 0.5f : 1.0f;
                xData[i * 2] = (float)(r * Math.cos(angle)) + rng.nextFloat() * 0.1f;
                xData[i * 2 + 1] = (float)(r * Math.sin(angle)) + rng.nextFloat() * 0.1f;
                yData[i] = (i < n / 2) ? 1f : 0f;
            }

            int hidden = 16;
            float[] w1Data = new float[2 * hidden];
            float[] w2Data = new float[hidden * 1];
            for (int i = 0; i < w1Data.length; i++) w1Data[i] = (rng.nextFloat() - 0.5f) * 0.1f;
            for (int i = 0; i < w2Data.length; i++) w2Data[i] = (rng.nextFloat() - 0.5f) * 0.1f;

            GpuAutogradTensor w1 = GpuAutogradTensor.fromHost(dev, w1Data, 2, hidden).requiresGrad(true);
            GpuAutogradTensor w2 = GpuAutogradTensor.fromHost(dev, w2Data, hidden, 1).requiresGrad(true);
            GpuAdam adam = new GpuAdam(dev, 0.1f);

            float initialLoss = Float.MAX_VALUE;
            for (int epoch = 0; epoch < 50; epoch++) {
                GpuAutogradTensor x = GpuAutogradTensor.fromHost(dev, xData, n, 2);
                GpuAutogradTensor y = GpuAutogradTensor.fromHost(dev, yData, n, 1);

                GpuAutogradTensor h = x.matmul(w1).relu();
                GpuAutogradTensor pred = h.matmul(w2).sigmoid();
                GpuAutogradTensor loss = pred.binaryCrossEntropy(y);

                float l = loss.data().toHost()[0];
                if (epoch == 0) initialLoss = l;

                loss.backward();
                adam.step(Arrays.asList(w1, w2));
                w1.zeroGrad();
                w2.zeroGrad();

                if (epoch == 49) {
                    assertTrue(l < initialLoss, "loss should decrease: " + l + " >= " + initialLoss);
                    assertTrue(l < 2f, "final loss should be low: " + l);
                }
            }
        }
    }

    @Test
    void sgdTrainsLinearBinaryClassifier() {
        try (CpuDevice dev = new CpuDevice()) {
            Random rng = new Random(123);
            int n = 16;
            float[] xData = new float[n * 2];
            float[] yData = new float[n];
            for (int i = 0; i < n; i++) {
                xData[i * 2] = rng.nextFloat() - 0.5f;
                xData[i * 2 + 1] = rng.nextFloat() - 0.5f;
                // Linearly separable: x < 0 -> class 1
                yData[i] = xData[i * 2] < 0f ? 1f : 0f;
            }

            float[] wData = new float[2];
            Arrays.fill(wData, 0.01f);
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev, wData, 2, 1).requiresGrad(true);
            GpuSgd sgd = new GpuSgd(dev, 0.5f);

            float initialLoss = Float.MAX_VALUE;
            for (int epoch = 0; epoch < 30; epoch++) {
                GpuAutogradTensor x = GpuAutogradTensor.fromHost(dev, xData, n, 2);
                GpuAutogradTensor y = GpuAutogradTensor.fromHost(dev, yData, n, 1);

                GpuAutogradTensor pred = x.matmul(w).sigmoid();
                GpuAutogradTensor loss = pred.binaryCrossEntropy(y);

                float l = loss.data().toHost()[0];
                if (epoch == 0) initialLoss = l;

                loss.backward();
                sgd.step(Arrays.asList(w));
                w.zeroGrad();

                if (epoch == 29) {
                    assertTrue(l < initialLoss, "loss should decrease: " + l + " >= " + initialLoss);
                    assertTrue(l < 5f, "final loss should be reasonable: " + l);
                }
            }
        }
    }

    @Test
    void mlpMultiClassOverfitsSpiral() {
        try (CpuDevice dev = new CpuDevice()) {
            Random rng = new Random(42);
            int n = 30; // 10 points per class
            int classes = 3;
            float[] xData = new float[n * 2];
            float[] yData = new float[n];
            for (int i = 0; i < n; i++) {
                int cls = i % classes;
                // Well-separated clusters
                xData[i * 2] = cls * 3f + rng.nextFloat() * 0.5f;
                xData[i * 2 + 1] = cls * 3f + rng.nextFloat() * 0.5f;
                yData[i] = (float) cls;
            }

            int hidden = 16;
            float[] w1Data = new float[2 * hidden];
            float[] w2Data = new float[hidden * classes];
            for (int i = 0; i < w1Data.length; i++) w1Data[i] = (rng.nextFloat() - 0.5f) * 0.5f;
            for (int i = 0; i < w2Data.length; i++) w2Data[i] = (rng.nextFloat() - 0.5f) * 0.5f;

            GpuAutogradTensor w1 = GpuAutogradTensor.fromHost(dev, w1Data, 2, hidden).requiresGrad(true);
            GpuAutogradTensor w2 = GpuAutogradTensor.fromHost(dev, w2Data, hidden, classes).requiresGrad(true);
            GpuAdam adam = new GpuAdam(dev, 0.05f);

            float initialLoss = Float.MAX_VALUE;
            for (int epoch = 0; epoch < 500; epoch++) {
                GpuAutogradTensor x = GpuAutogradTensor.fromHost(dev, xData, n, 2);
                GpuAutogradTensor y = GpuAutogradTensor.fromHost(dev, yData, n);

                GpuAutogradTensor h = x.matmul(w1).relu();
                GpuAutogradTensor logits = h.matmul(w2);
                GpuAutogradTensor loss = logits.crossEntropy(y);

                float l = loss.data().toHost()[0];
                if (epoch == 0) initialLoss = l;

                loss.backward();
                adam.step(Arrays.asList(w1, w2));
                w1.zeroGrad();
                w2.zeroGrad();

                if (epoch == 499) {
                    assertTrue(l < initialLoss, "loss should decrease: " + l + " >= " + initialLoss);
                    assertTrue(l < 15f, "final loss should be low: " + l);
                }
            }
        }
    }
}
