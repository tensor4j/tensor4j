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

import java.nio.file.Path;
import java.util.Locale;

/** Checkpoint format for tinygrad-track weight bundles. */
public enum WeightFormat {

    /** tensor4j JSON bundle ({@code .t4j.json}). */
    T4J_JSON,

    /** Hugging Face safetensors ({@code .safetensors}) — tinygrad {@code nn.state.safe_save}. */
    SAFETENSORS;

    public static WeightFormat fromSystemProperty() {
        String value = System.getProperty("tensor4j.weight.format", "safetensors");
        if ("t4j_json".equalsIgnoreCase(value) || "json".equalsIgnoreCase(value)) {
            return T4J_JSON;
        }
        if ("safetensors".equalsIgnoreCase(value) || "safe".equalsIgnoreCase(value)) {
            return SAFETENSORS;
        }
        return SAFETENSORS;
    }

    public static WeightFormat fromPath(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".safetensors")) {
            return SAFETENSORS;
        }
        if (name.endsWith(".t4j.json") || name.endsWith(".json")) {
            return T4J_JSON;
        }
        return fromSystemProperty();
    }
}
