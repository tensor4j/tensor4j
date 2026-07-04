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

import com.github.tensor4j.io.ModelLoader;
import com.github.tensor4j.io.WeightFormat;
import java.nio.file.Files;
import java.nio.file.Path;

/** Convert bundled .t4j.json weights to .safetensors (same tensor keys). */
public final class ConvertWeights {

    private ConvertWeights() {
    }

    public static void main(String[] args) throws Exception {
        Path in = Path.of(args.length > 0 ? args[0] : "java/resources/models/algebra-v1.t4j.json");
        Path out = Path.of(args.length > 1 ? args[1] : defaultSafetensorsPath(in));
        Files.createDirectories(out.getParent());
        ModelLoader.save(out, ModelLoader.load(in), WeightFormat.SAFETENSORS);
        System.out.println("converted " + in.toAbsolutePath() + " -> " + out.toAbsolutePath());
    }

    private static String defaultSafetensorsPath(Path in) {
        String name = in.getFileName().toString();
        if (name.endsWith(".t4j.json")) {
            return in.getParent().resolve(name.replace(".t4j.json", ".safetensors")).toString();
        }
        return in.resolveSibling(in.getFileName() + ".safetensors").toString();
    }
}
