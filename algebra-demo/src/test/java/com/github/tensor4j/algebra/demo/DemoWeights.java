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

import com.github.tensor4j.io.ModelLoader;
import com.github.tensor4j.io.Safetensors;
import com.github.tensor4j.models.algebra.AlgebraModel;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import com.github.tensor4j.core.Tensor;

/** Loads demo weights bundled under {@code src/test/resources/models/}. */
final class DemoWeights {

    static final String SAFETENSORS_RESOURCE = "/models/algebra-v1.safetensors";

    private DemoWeights() {
    }

    static void loadInto(AlgebraModel model) throws IOException {
        try (InputStream in = DemoWeights.class.getResourceAsStream(SAFETENSORS_RESOURCE)) {
            if (in == null) {
                throw new IOException("demo weights not found: " + SAFETENSORS_RESOURCE);
            }
            Map<String, Tensor> tensors = Safetensors.load(in.readAllBytes());
            ModelLoader.applyToSequential(model.network(), tensors);
        }
    }
}
