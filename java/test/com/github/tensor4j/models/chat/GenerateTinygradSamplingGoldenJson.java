/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat;

import com.github.tensor4j.models.chat.reference.TinygradSamplingGoldenCase;
import com.github.tensor4j.models.chat.reference.TinygradSamplingGoldenCases;
import java.nio.file.Files;
import java.nio.file.Path;

/** Regenerates {@code java/test/resources/tinygrad-sampling-golden.json}. */
public final class GenerateTinygradSamplingGoldenJson {

    private GenerateTinygradSamplingGoldenJson() {
    }

    public static void main(String[] args) throws Exception {
        StringBuilder json = new StringBuilder("{\n  \"cases\": [\n");
        TinygradSamplingGoldenCase[] cases = TinygradSamplingGoldenCases.all();
        for (int i = 0; i < cases.length; i++) {
            TinygradSamplingGoldenCase golden = cases[i];
            json.append("    {\n");
            json.append("      \"name\": \"").append(golden.name()).append("\",\n");
            json.append("      \"logits\": ").append(floatArray(golden.logits())).append(",\n");
            json.append("      \"temperature\": ").append(golden.temperature()).append(",\n");
            json.append("      \"seed\": ").append(golden.seed()).append(",\n");
            json.append("      \"multinomialRoll\": ").append(golden.multinomialRoll()).append(",\n");
            json.append("      \"topK\": ").append(golden.topK()).append(",\n");
            json.append("      \"topP\": ").append(golden.topP()).append(",\n");
            json.append("      \"alphaFrequency\": ").append(golden.alphaFrequency()).append(",\n");
            json.append("      \"alphaPresence\": ").append(golden.alphaPresence()).append(",\n");
            json.append("      \"alphaCounts\": ").append(intArray(golden.alphaCounts())).append(",\n");
            json.append("      \"gumbelMax\": ").append(golden.gumbelMax()).append(",\n");
            json.append("      \"expectedToken\": ").append(golden.expectedToken()).append("\n");
            json.append("    }");
            if (i + 1 < cases.length) {
                json.append(',');
            }
            json.append('\n');
        }
        json.append("  ]\n}\n");
        Path out = Path.of("java/resources/tinygrad-sampling-golden.json");
        Files.createDirectories(out.getParent());
        Files.writeString(out, json.toString());
        System.out.println("wrote " + cases.length + " cases to " + out.toAbsolutePath());
    }

    private static String floatArray(float[] values) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(values[i]);
        }
        return sb.append(']').toString();
    }

    private static String intArray(int[] values) {
        if (values == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(values[i]);
        }
        return sb.append(']').toString();
    }
}
