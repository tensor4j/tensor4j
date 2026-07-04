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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import com.github.tensor4j.io.ModelLoader;
import com.github.tensor4j.io.WeightFormat;
import com.github.tensor4j.models.algebra.AlgebraModel;
import com.github.tensor4j.models.algebra.AlgebraModel.AlgebraResult;
import com.github.tensor4j.models.algebra.AlgebraTrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Demo: train in Java, export safetensors, reload into a fresh model, infer.
 * Run via {@code mvn verify}.
 */
class TrainExportRoundTripDemoIT {

    private static final float MAX_ABS_ERROR = 1.0f;

    @Test
    void trainExportSafetensorsAndInfer(@TempDir Path temp) throws Exception {
        AlgebraDemoReporter.banner("train -> safetensors export -> reload -> infer");

        AlgebraModel trained = new AlgebraModel();
        AlgebraTrainer trainer = new AlgebraTrainer(trained);
        List<Float> losses = trainer.train(400, 0.05f, 32);
        System.out.printf("  training: epochs=400  initial loss=%.4f  final loss=%.4f%n",
                losses.get(0), losses.get(losses.size() - 1));
        assertTrue(losses.get(losses.size() - 1) < losses.get(0), "loss should decrease during training");

        Path weights = temp.resolve("algebra-demo.safetensors");
        ModelLoader.save(weights, ModelLoader.exportTensors(trained.network()), WeightFormat.SAFETENSORS);
        System.out.println("  exported: " + weights.toAbsolutePath());

        AlgebraModel reloaded = new AlgebraModel();
        reloaded.loadWeights(weights);
        System.out.println("  reloaded weights into fresh AlgebraModel");

        List<String> equations = DemoEquations.load();
        int passed = 0;
        for (int i = 0; i < equations.size(); i++) {
            AlgebraResult result = reloaded.infer(equations.get(i));
            AlgebraDemoReporter.result(result);
            if (result.error() <= MAX_ABS_ERROR) {
                passed++;
            }
        }
        assertTrue(Files.size(weights) > 64, "safetensors file should contain header + tensor bytes");
        assertTrue(passed >= equations.size() / 2, "expected most demo equations within tolerance");
        AlgebraDemoReporter.summary(passed, equations.size(), MAX_ABS_ERROR);
    }
}
