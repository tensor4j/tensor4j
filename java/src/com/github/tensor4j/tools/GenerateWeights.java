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

import java.nio.file.Files;
import java.nio.file.Path;
import com.github.tensor4j.io.ModelLoader;
import com.github.tensor4j.io.WeightFormat;
import com.github.tensor4j.models.algebra.AlgebraModel;
import com.github.tensor4j.models.algebra.AlgebraTrainer;

/** Dev utility: train algebra head and emit tinygrad-compatible weights. */
public final class GenerateWeights {

    private GenerateWeights() {
    }

    public static void main(String[] args) throws Exception {
        Path out = Path.of(args.length > 0 ? args[0] : "java/resources/models/algebra-v1.safetensors");
        WeightFormat format = WeightFormat.fromPath(out);
        AlgebraModel model = new AlgebraModel();
        new AlgebraTrainer(model).train(400, 0.05f, 32);
        Files.createDirectories(out.getParent());
        ModelLoader.save(out, ModelLoader.exportTensors(model.network()), format);
        System.out.println("wrote " + format + " weights to " + out.toAbsolutePath());
    }
}
