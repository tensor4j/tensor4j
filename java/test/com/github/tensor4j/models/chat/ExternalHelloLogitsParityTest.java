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

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.reference.LogitsTopK;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * Compares Java Hello prefill top-10 logits vs tinygrad capture.
 *
 * <p>Regenerate golden: {@code python tools/capture_tinygrad_hello_logits.py}
 * Requires {@code TENSOR4J_GGUF_PATH} pointing at the same GGUF file used for capture.
 */
@EnabledIfEnvironmentVariable(named = "TENSOR4J_GGUF_PATH", matches = ".+")
class ExternalHelloLogitsParityTest {

    private static final String FIXTURE = "/fixtures/tinygrad-hello-logits.json";
    private static final Pattern TOKEN_ID = Pattern.compile("\"token_id\"\\s*:\\s*(\\d+)");
    private static final Pattern LOGIT = Pattern.compile("\"logit\"\\s*:\\s*([0-9.eE+-]+)");
    private static final float TOL = 0.05f;

    @Test
    void helloPrefillTop10MatchesTinygradCapture() throws Exception {
        String json = readFixture();
        if (json == null) {
            org.junit.jupiter.api.Assumptions.assumeTrue(
                    false, "missing " + FIXTURE + " — run tools/capture_tinygrad_hello_logits.py");
        }
        float[] expected = parseGoldenLogits(json);
        int[] expectedIds = parseGoldenTokenIds(json);

        Path path = Paths.get(System.getenv("TENSOR4J_GGUF_PATH"));
        try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
            ChatModel model = ChatModel.fromGguf(mapped);
            int[] tokens = model.tokenizer().encode("Hello");
            model.resetCache();
            float[] actual = model.forward(tokens);

            for (int i = 0; i < expectedIds.length; i++) {
                int id = expectedIds[i];
                float diff = Math.abs(expected[i] - actual[id]);
                assertTrue(
                        diff <= TOL,
                        "token " + id + " logit diff " + diff + " expected=" + expected[i]
                                + " actual=" + actual[id] + "\n"
                                + LogitsTopK.formatReport("java", actual, model.tokenizer(), 10));
            }
        }
    }

    private static String readFixture() {
        try (var in = ExternalHelloLogitsParityTest.class.getResourceAsStream(FIXTURE)) {
            if (in == null) {
                return null;
            }
            return new String(in.readAllBytes());
        } catch (Exception ex) {
            throw new IllegalStateException("failed to read " + FIXTURE, ex);
        }
    }

    private static float[] parseGoldenLogits(String json) {
        Matcher m = LOGIT.matcher(json);
        java.util.ArrayList<Float> values = new java.util.ArrayList<>();
        while (m.find()) {
            values.add(Float.parseFloat(m.group(1)));
        }
        float[] out = new float[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static int[] parseGoldenTokenIds(String json) {
        Matcher m = TOKEN_ID.matcher(json);
        java.util.ArrayList<Integer> values = new java.util.ArrayList<>();
        while (m.find()) {
            values.add(Integer.parseInt(m.group(1)));
        }
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
    }
}
