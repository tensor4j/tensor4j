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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.nn.Linear;
import com.github.tensor4j.nn.Sequential;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** tinygrad {@code nn.state.safe_load} / {@code safe_save} parity (F32). */
class SafetensorsTest {

    @Test
    void encodeAndLoadRoundTrip(@TempDir Path temp) throws Exception {
        Sequential network = sampleNetwork();
        Map<String, Tensor> original = ModelLoader.exportTensors(network);

        Path file = temp.resolve("weights.safetensors");
        Safetensors.save(file, original);
        Map<String, Tensor> loaded = Safetensors.load(file);

        assertEquals(original.size(), loaded.size());
        for (Map.Entry<String, Tensor> entry : original.entrySet()) {
            assertAllClose(entry.getValue(), loaded.get(entry.getKey()), 1e-6f);
        }
    }

    @Test
    void inMemoryRoundTripMatchesNetworkWeights() {
        Sequential network = sampleNetwork();
        Map<String, Tensor> original = ModelLoader.exportTensors(network);
        Map<String, Tensor> loaded = Safetensors.load(Safetensors.encode(original));
        ModelLoader.applyToSequential(network, loaded);
        assertAllClose(original.get("fc1.weight"), network.findLinear("fc1").weight(), 1e-6f);
    }

    @Test
    void bundledJsonMatchesSafetensorsEncoding() throws Exception {
        Map<String, Tensor> json = ModelLoader.loadResource("/models/algebra-v1.t4j.json");
        Map<String, Tensor> fromSafe = Safetensors.load(Safetensors.encode(json));
        assertEquals(json.size(), fromSafe.size());
        for (Map.Entry<String, Tensor> entry : json.entrySet()) {
            assertAllClose(entry.getValue(), fromSafe.get(entry.getKey()), 1e-5f);
        }
    }

    @Test
    void headerIsEightBytePadded(@TempDir Path temp) throws Exception {
        Map<String, Tensor> tensors = Map.of("x", Tensor.of(new float[] {1f, 2f}, 2));
        byte[] bytes = Safetensors.encode(tensors);
        long headerLen = java.nio.ByteBuffer.wrap(bytes, 0, 8)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .getLong();
        assertTrue(headerLen % 8 == 0, "safetensors header must be 8-byte aligned");
        assertEquals(8 + headerLen + 8, bytes.length);
    }

    private static Sequential sampleNetwork() {
        return new Sequential()
                .add("fc1", new Linear(3, 4, "fc1"))
                .add("fc2", new Linear(4, 1, "fc2"));
    }
}
