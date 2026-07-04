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

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.support.ParityFixtureLoader;
import org.junit.jupiter.api.Test;

/** Layer 3b: domain numeric specs aligned with tinygrad parity fixtures. */
class AlgebraNumericTest {

    @Test
    void normalizedFeaturesMatchParityFixture() {
        ParityFixtureLoader.ParityCase parityCase = ParityFixtureLoader.loadResource("/parity/tinygrad-ops.json")
                .stream()
                .filter(entry -> "algebra_normalized_features".equals(entry.name()))
                .findFirst()
                .orElseThrow();
        AlgebraEquation equation = new AlgebraEquation(2f, 3f, 11f, "2x+3=11");
        assertAllClose(parityCase.input().data(), equation.features());
        assertAllClose(parityCase.expected().data(), equation.normalizedFeatures());
    }

    @Test
    void exactSolutionIsStableFloatDivision() {
        AlgebraEquation equation = AlgebraEquation.parse("2x + 3 = 11");
        assertEquals(4f, equation.exactSolution(), 1e-5f);
    }

    @Test
    void featureVectorUsesFlatBufferNotNestedRows() {
        AlgebraEquation equation = AlgebraEquation.parse("-3x - 5 = 7");
        float[] features = equation.normalizedFeatures();
        assertEquals(3, features.length);
        assertAllClose(new float[] {-0.3f, -0.5f, 0.7f}, features, 1e-5f);
    }
}
