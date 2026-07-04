/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.algebra;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.optim.Sgd;
import org.junit.jupiter.api.Test;

class MlpAlgebraFitTest {

    @Test
    void mlpOverfitsSingleEquation() {
        AlgebraModel model = new AlgebraModel();
        AlgebraEquation equation = AlgebraEquation.parse("2x + 3 = 11");
        Tensor input = Tensor.of(equation.normalizedFeatures(), 1, 3);
        Tensor target = Tensor.of(new float[] {equation.exactSolution()}, 1, 1);
        Sgd optimizer = new Sgd(0.05f);
        for (int step = 0; step < 400; step++) {
            model.network().zeroGrad();
            Tensor prediction = model.network().forward(input);
            Tensor diff = prediction.sub(target);
            Tensor loss = diff.mul(diff).mean();
            loss.backward();
            optimizer.step(model.network().parameters());
        }
        float predicted = model.predict(equation);
        assertFalse(Float.isNaN(predicted));
        assertTrue(Math.abs(predicted - equation.exactSolution()) < 0.5f);
    }
}
