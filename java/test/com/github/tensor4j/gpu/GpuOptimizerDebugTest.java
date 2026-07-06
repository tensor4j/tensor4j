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

import org.junit.jupiter.api.Test;
import com.github.tensor4j.gpu.core.GpuAutogradTensor;
import com.github.tensor4j.gpu.core.GpuSgd;
import com.github.tensor4j.gpu.core.GpuTensorMath;
import com.github.tensor4j.gpu.ref.CpuDevice;

class GpuOptimizerDebugTest {

    @Test
    void sgdOneStep() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuAutogradTensor w = GpuAutogradTensor.fromHost(dev, new float[]{3f, 4f}, 2).requiresGrad(true);
            GpuAutogradTensor target = GpuAutogradTensor.fromHost(dev, new float[]{1f, 2f}, 2);

            GpuAutogradTensor diff = w.sub(target);
            GpuAutogradTensor sq = diff.mul(diff);
            GpuAutogradTensor loss = sq.sum();
            assertEquals(8f, sq.toHost()[0] + sq.toHost()[1], 1e-5f);
            loss.backward();

            assertNotNull(w.grad());
            float[] g = w.grad().toHost();
            float gradExpected0 = 2f * (3f - 1f); // ∂loss/∂w0 = 2*(3-1) = 4
            float gradExpected1 = 2f * (4f - 2f); // ∂loss/∂w1 = 2*(4-2) = 4
            assertEquals(gradExpected0, g[0], 1e-5f, "grad[0]");
            assertEquals(gradExpected1, g[1], 1e-5f, "grad[1]");

            GpuSgd opt = new GpuSgd(dev, 0.1f);
            opt.step(Arrays.asList(w));

            float[] wNew = w.toHost();
            assertEquals(3f - 0.1f * 4f, wNew[0], 1e-5f, "w[0] after step");
            assertEquals(4f - 0.1f * 4f, wNew[1], 1e-5f, "w[1] after step");
        }
    }
}
