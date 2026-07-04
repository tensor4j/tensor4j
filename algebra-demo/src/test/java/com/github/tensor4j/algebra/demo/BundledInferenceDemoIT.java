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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import com.github.tensor4j.models.algebra.AlgebraModel;
import com.github.tensor4j.models.algebra.AlgebraModel.AlgebraResult;
import org.junit.jupiter.api.Test;

/**
 * Demo: load bundled safetensors weights and infer {@code x} for a fixed equation set.
 * Run via {@code mvn verify}.
 */
class BundledInferenceDemoIT {

    private static final float MAX_ABS_ERROR = 1.0f;

    @Test
    void bundledSafetensorsSolvesDemoEquations() throws Exception {
        AlgebraDemoReporter.banner("bundled safetensors inference");
        AlgebraModel model = new AlgebraModel();
        DemoWeights.loadInto(model);

        List<String> equations = DemoEquations.load();
        int passed = 0;
        for (int i = 0; i < equations.size(); i++) {
            AlgebraResult result = model.infer(equations.get(i));
            AlgebraDemoReporter.result(result);
            if (result.error() <= MAX_ABS_ERROR) {
                passed++;
            }
        }
        assertTrue(passed >= equations.size() / 2, "expected most demo equations within tolerance");
        AlgebraDemoReporter.summary(passed, equations.size(), MAX_ABS_ERROR);
    }
}
