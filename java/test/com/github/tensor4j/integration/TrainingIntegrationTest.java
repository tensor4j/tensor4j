/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.integration;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import com.github.tensor4j.models.algebra.AlgebraModel;
import com.github.tensor4j.models.algebra.AlgebraTrainer;
import org.junit.jupiter.api.Test;

/** Layer 4: training-loop integration. */
class TrainingIntegrationTest {

    @Test
    void algebraLossDecreasesOverEpochs() {
        AlgebraModel model = new AlgebraModel();
        AlgebraTrainer trainer = new AlgebraTrainer(model);
        List<Float> losses = trainer.train(200, 0.05f, 32);
        float first = losses.get(0);
        float last = losses.get(losses.size() - 1);
        assertTrue(last < first, "final loss should be lower than initial loss: " + first + " vs " + last);
        assertTrue(last < 50f, "loss should decrease substantially, last=" + last);
    }
}
