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

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import com.github.tensor4j.gpu.core.GpuAdam;
import com.github.tensor4j.gpu.core.GpuAutogradTensor;
import com.github.tensor4j.gpu.core.GpuSgd;
import com.github.tensor4j.gpu.core.GpuTensor;
import com.github.tensor4j.gpu.core.GpuTensorMath;
import com.github.tensor4j.gpu.ref.CpuDevice;

/**
 * Tests for GPU optimizers (SGD, Adam).
 */
class GpuOptimizerTest {

    @Test
    void sgdStepsDecreasesLoss() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev, new float[]{3f, 4f}, 2).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f}, 2);
            GpuSgd opt = new GpuSgd(dev, 0.1f);
            List<GpuAutogradTensor> params = Arrays.asList(w);

            float lossBefore = computeMse(w, target, dev);
            for (int i = 0; i < 50; i++) {
                GpuAutogradTensor loss = computeMseGraph(w, target, dev);
                loss.backward();
                opt.step(params);
                opt.zeroGrad(params);
            }
            float lossAfter = computeMse(w, target, dev);
            assertTrue(lossAfter < lossBefore, "loss should decrease after SGD steps");
        }
    }

    @Test
    void sgdConvergesToTarget() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev, new float[]{0f}, 1).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev, new float[]{5f}, 1);
            GpuSgd opt = new GpuSgd(dev, 0.1f);
            List<GpuAutogradTensor> params = Arrays.asList(w);

            for (int i = 0; i < 200; i++) {
                GpuAutogradTensor diff = w.sub(target);
                GpuAutogradTensor loss = diff.mul(diff);
                loss.backward();
                opt.step(params);
                opt.zeroGrad(params);
            }
            float val = w.toHost()[0];
            assertEquals(5f, val, 0.1f, "SGD should converge w toward target");
        }
    }

    @Test
    void sgdWeightDecay() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev, new float[]{10f}, 1).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev, new float[]{0f}, 1);
            GpuSgd opt = new GpuSgd(dev, 0.1f, 0.01f, 0f);
            List<GpuAutogradTensor> params = Arrays.asList(w);

            for (int i = 0; i < 100; i++) {
                GpuAutogradTensor diff = w.sub(target);
                GpuAutogradTensor loss = diff.mul(diff);
                loss.backward();
                opt.step(params);
                opt.zeroGrad(params);
            }
            float val = w.toHost()[0];
            assertTrue(Math.abs(val) < 10f, "weight decay should shrink w toward 0");
        }
    }

    @Test
    void adamStepsDecreasesLoss() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev, new float[]{3f, 4f}, 2).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f}, 2);
            GpuAdam opt = new GpuAdam(dev, 0.1f);
            List<GpuAutogradTensor> params = Arrays.asList(w);

            float lossBefore = computeMse(w, target, dev);
            for (int i = 0; i < 50; i++) {
                GpuAutogradTensor loss = computeMseGraph(w, target, dev);
                loss.backward();
                opt.step(params);
                opt.zeroGrad(params);
            }
            float lossAfter = computeMse(w, target, dev);
            assertTrue(lossAfter < lossBefore, "loss should decrease after Adam steps");
        }
    }

    @Test
    void adamConvergesToTarget() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev, new float[]{0f}, 1).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev, new float[]{5f}, 1);
            GpuAdam opt = new GpuAdam(dev, 0.1f);
            List<GpuAutogradTensor> params = Arrays.asList(w);

            for (int i = 0; i < 200; i++) {
                GpuAutogradTensor diff = w.sub(target);
                GpuAutogradTensor loss = diff.mul(diff);
                loss.backward();
                opt.step(params);
                opt.zeroGrad(params);
            }
            float val = w.toHost()[0];
            assertEquals(5f, val, 0.5f, "Adam should converge w toward target");
        }
    }

    @Test
    void adamWeightDecay() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev, new float[]{10f}, 1).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev, new float[]{0f}, 1);
            GpuAdam opt = new GpuAdam(dev, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f);
            List<GpuAutogradTensor> params = Arrays.asList(w);

            for (int i = 0; i < 100; i++) {
                GpuAutogradTensor diff = w.sub(target);
                GpuAutogradTensor loss = diff.mul(diff);
                loss.backward();
                opt.step(params);
                opt.zeroGrad(params);
            }
            float val = w.toHost()[0];
            assertTrue(Math.abs(val) < 10f, "Adam weight decay should shrink w toward 0");
        }
    }

    @Test
    void sgdMultipleParameters() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor w1 = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f}, 2).requiresGrad(true);
            GpuAutogradTensor w2 = GpuAutogradTensor.fromHost(dev, new float[]{3f, 4f}, 2).requiresGrad(true);
            GpuSgd opt = new GpuSgd(dev, 0.1f);
            List<GpuAutogradTensor> params = Arrays.asList(w1, w2);

            for (int i = 0; i < 10; i++) {
                GpuAutogradTensor s = w1.add(w2);
                GpuAutogradTensor loss = s.mul(s);
                loss.backward();
                opt.step(params);
                opt.zeroGrad(params);
            }
            float[] v1 = w1.toHost();
            float[] v2 = w2.toHost();
            for (float v : v1) {
                assertTrue(v < 1f, "sgd multi-param should shrink w1");
            }
            for (float v : v2) {
                assertTrue(v < 3f, "sgd multi-param should shrink w2");
            }
        }
    }

    private static float computeMse(GpuAutogradTensor w, GpuAutogradTensor target, CpuDevice dev) {
        GpuTensorMath math = new GpuTensorMath(dev);
        GpuTensor diff = math.sub(w.data(), target.data());
        GpuTensor sq = math.mul(diff, diff);
        GpuTensor sum = math.sum(sq);
        float[] h = sum.toHost();
        return h[0] / w.numel();
    }

    private static GpuAutogradTensor computeMseGraph(GpuAutogradTensor w, GpuAutogradTensor target, CpuDevice dev) {
        GpuAutogradTensor diff = w.sub(target);
        GpuAutogradTensor sq = diff.mul(diff);
        return sq.sum();
    }
}
