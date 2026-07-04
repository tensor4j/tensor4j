/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.io.ModelLoader;
import com.github.tensor4j.io.ModelState;
import com.github.tensor4j.models.algebra.AlgebraModel;
import com.github.tensor4j.models.algebra.AlgebraTrainer;
import com.github.tensor4j.nn.MlpBuilder;
import com.github.tensor4j.nn.Sequential;

/** tinygrad-style {@code safe_save(get_state_dict(model))} with optional algebra training profile. */
public final class GenerateWeightsRunner {

    public GenerateWeightsResult run(GenerateWeightsOptions options) throws IOException {
        Sequential network = buildNetwork(options);
        Map<String, Tensor> stateDict = ModelState.getStateDict(network);
        Path parent = options.output().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        ModelLoader.save(options.output(), stateDict, options.format());
        return new GenerateWeightsResult(options.profile(), options.output(), options.format(), stateDict.size());
    }

    private static Sequential buildNetwork(GenerateWeightsOptions options) {
        if (options.profile() == WeightProfile.ALGEBRA) {
            AlgebraModel model = new AlgebraModel();
            new AlgebraTrainer(model).train(options.epochs(), options.learningRate(), options.batchSize());
            return model.network();
        }
        return MlpBuilder.fullyConnected(options.layerSizes());
    }

    public record GenerateWeightsResult(
            WeightProfile profile,
            Path output,
            com.github.tensor4j.io.WeightFormat format,
            int tensorCount) {

        public String summary() {
            return "profile=" + profile + " format=" + format + " tensors=" + tensorCount + " path="
                    + output.toAbsolutePath();
        }
    }
}
