/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.io;

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.nn.Linear;
import com.github.tensor4j.nn.Sequential;
import org.junit.jupiter.api.Test;

/** tinygrad-compatible weight bundle round-trip. */
class ModelLoaderTest {

    @Test
    void exportAndParsePreservesTensorValues() {
        Sequential network = new Sequential()
                .add("fc1", new Linear(3, 4, "fc1"))
                .add("fc2", new Linear(4, 1, "fc2"));
        String json = ModelLoader.export(network);
        Map<String, Tensor> parsed = ModelLoader.parse(json);
        assertEquals(4, parsed.size());
        assertTrue(parsed.containsKey("fc1.weight"));
        assertAllClose(network.findLinear("fc1").weight(), parsed.get("fc1.weight"), 1e-6f);
        ModelLoader.applyToSequential(network, parsed);
        assertAllClose(network.findLinear("fc1").weight(), parsed.get("fc1.weight"), 1e-6f);
    }

    @Test
    void saveAndLoadSafetensorsViaModelLoader(@org.junit.jupiter.api.io.TempDir java.nio.file.Path temp)
            throws Exception {
        Sequential network = new Sequential()
                .add("fc1", new Linear(2, 3, "fc1"));
        Map<String, Tensor> tensors = ModelLoader.exportTensors(network);
        java.nio.file.Path path = temp.resolve("m.safetensors");
        ModelLoader.save(path, tensors, WeightFormat.SAFETENSORS);
        Map<String, Tensor> loaded = ModelLoader.load(path);
        assertAllClose(tensors.get("fc1.weight"), loaded.get("fc1.weight"), 1e-6f);
        assertAllClose(tensors.get("fc1.bias"), loaded.get("fc1.bias"), 1e-6f);
    }

    @Test
    void fromPathDetectsSafetensorsExtension() {
        assertEquals(WeightFormat.SAFETENSORS, WeightFormat.fromPath(java.nio.file.Path.of("a.safetensors")));
        assertEquals(WeightFormat.T4J_JSON, WeightFormat.fromPath(java.nio.file.Path.of("a.t4j.json")));
    }

    @Test
    void systemPropertyDefaultsToSafetensors() {
        String key = "tensor4j.weight.format";
        String previous = System.getProperty(key);
        try {
            System.clearProperty(key);
            assertEquals(WeightFormat.SAFETENSORS, WeightFormat.fromSystemProperty());
            System.setProperty(key, "t4j_json");
            assertEquals(WeightFormat.T4J_JSON, WeightFormat.fromSystemProperty());
        } finally {
            if (previous == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, previous);
            }
        }
    }
}
