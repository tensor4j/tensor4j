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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import com.github.tensor4j.io.ModelLoader;
import com.github.tensor4j.io.WeightFormat;
import com.github.tensor4j.io.ModelState;
import com.github.tensor4j.nn.MlpBuilder;
import com.github.tensor4j.nn.Sequential;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GenerateWeightsRunnerTest {

    @Test
    void generalProfileExportsStateDict(@TempDir Path temp) throws Exception {
        Path out = temp.resolve("model.safetensors");
        GenerateWeightsOptions options = GenerateWeightsOptions.parse(new String[] {
            "--layers", "3,4,1",
            "--out", out.toString(),
        });
        assertEquals(WeightProfile.GENERAL, options.profile());

        GenerateWeightsRunner.GenerateWeightsResult result = new GenerateWeightsRunner().run(options);
        assertEquals(WeightProfile.GENERAL, result.profile());
        assertEquals(4, result.tensorCount());

        Map<String, com.github.tensor4j.core.Tensor> loaded = ModelLoader.load(out);
        assertTrue(loaded.containsKey("fc1.weight"));
        assertTrue(loaded.containsKey("fc2.bias"));

        Sequential network = MlpBuilder.fullyConnected(new int[] {3, 4, 1});
        ModelState.loadStateDict(network, loaded);
        assertEquals(3, network.findLinear("fc1").weight().shape().dim(0));
    }

    @Test
    void defaultsToGeneralProfile() {
        GenerateWeightsOptions options = GenerateWeightsOptions.parse(new String[] {});
        assertEquals(WeightProfile.GENERAL, options.profile());
        assertEquals(WeightFormat.SAFETENSORS, options.format());
    }

    @Test
    void algebraProfileParsesExplicitly() {
        GenerateWeightsOptions options = GenerateWeightsOptions.parse(new String[] {
            "--profile", "algebra",
            "--epochs", "10",
        });
        assertEquals(WeightProfile.ALGEBRA, options.profile());
        assertEquals(10, options.epochs());
    }
}
