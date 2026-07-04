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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.optim.Sgd;
import org.junit.jupiter.api.Test;

/** Layer 3: domain model behavior. */
class AlgebraModelTest {

    @Test
    void parseSimpleLinearEquation() {
        AlgebraEquation eq = AlgebraEquation.parse("2x + 3 = 11");
        assertEquals(2f, eq.a(), 1e-5f);
        assertEquals(3f, eq.b(), 1e-5f);
        assertEquals(11f, eq.c(), 1e-5f);
        assertEquals(4f, eq.exactSolution(), 1e-5f);
    }

    @Test
    void parseImplicitCoefficientOne() {
        AlgebraEquation eq = AlgebraEquation.parse("x + 5 = 12");
        assertEquals(1f, eq.a(), 1e-5f);
        assertEquals(5f, eq.b(), 1e-5f);
        assertEquals(7f, eq.exactSolution(), 1e-5f);
    }

    @Test
    void rejectZeroCoefficient() {
        assertThrows(IllegalArgumentException.class, new ParseEquationAction("0x + 1 = 2"));
    }

    @Test
    void trainedModelPredictsWithinTolerance() {
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
        AlgebraModel.AlgebraResult result = model.infer("2x + 3 = 11");
        assertEquals(4f, result.exact(), 1e-5f);
        assertTrue(result.error() < 0.5f, "predicted x should be near exact solution, error=" + result.error());
    }

    private static final class ParseEquationAction implements org.junit.jupiter.api.function.Executable {
        private final String equation;

        ParseEquationAction(String equation) {
            this.equation = equation;
        }

        @Override
        public void execute() {
            AlgebraEquation.parse(equation);
        }
    }
}
