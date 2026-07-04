/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.algebra.demo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.models.algebra.AlgebraEquation;
import org.junit.jupiter.api.Test;

class WorkedSolutionTest {

    @Test
    void stepsForTwoXPlusThreeEqualsEleven() {
        AlgebraEquation equation = AlgebraEquation.parse("2x + 3 = 11");
        assertEquals(
                "2x = 11 - 3 = 8",
                WorkedSolution.steps(equation).get(1));
        assertEquals(
                "x = 8 / 2 = 4",
                WorkedSolution.steps(equation).get(2));
    }

    @Test
    void stepsForFourXEqualsTwenty() {
        AlgebraEquation equation = AlgebraEquation.parse("4x = 20");
        assertEquals("4x = 20", WorkedSolution.steps(equation).get(0));
        assertEquals("x = 20 / 4 = 5", WorkedSolution.steps(equation).get(1));
    }
}
