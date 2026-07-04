/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.nn;

import static com.github.tensor4j.support.TensorAssert.assertClose;
import static com.github.tensor4j.support.TensorAssert.assertNoNaN;

import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.optim.Sgd;
import org.junit.jupiter.api.Test;

class LinearModuleTest {

    @Test
    void learnsScalarMultiple() {
        Linear linear = new Linear(1, 1, "probe");
        Sgd optimizer = new Sgd(0.05f);
        for (int step = 0; step < 200; step++) {
            float value = step % 5 + 1f;
            Tensor input = Tensor.of(new float[] {value}, 1, 1);
            Tensor target = Tensor.of(new float[] {3f * value}, 1, 1);
            linear.zeroGrad();
            Tensor prediction = linear.forward(input);
            Tensor diff = prediction.sub(target);
            Tensor loss = diff.mul(diff).mean();
            loss.backward();
            optimizer.step(linear.parameters());
        }
        Tensor probe = linear.forward(Tensor.of(new float[] {4f}, 1, 1));
        assertNoNaN(probe);
        assertClose(12f, probe.data()[0], 0.75f);
    }
}
