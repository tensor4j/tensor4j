/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.reference;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Loads checked-in tinygrad Tensor runtime captures (no Python/tinygrad at test time).
 *
 * <p>Regenerate via {@code python scripts/capture_tinygrad_runtime_golden.py}.
 */
public final class TinygradRuntimeGoldenLoader {

    private static final Pattern NAME_MARKER = Pattern.compile("\\{\\s*\"name\":\\s*\"([^\"]+)\"");

    private TinygradRuntimeGoldenLoader() {
    }

    public static List<TinygradRuntimeGoldenCase> load() {
        try (InputStream in = TinygradRuntimeGoldenLoader.class.getResourceAsStream("/tinygrad-runtime-sampling-golden.json")) {
            if (in == null) {
                throw new IllegalStateException("missing tinygrad-runtime-sampling-golden.json on classpath");
            }
            String json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return parse(json);
        } catch (Exception e) {
            throw new IllegalStateException("failed to load tinygrad runtime golden", e);
        }
    }

    public static TinygradRuntimeGoldenCase require(String name) {
        for (TinygradRuntimeGoldenCase golden : load()) {
            if (golden.name().equals(name)) {
                return golden;
            }
        }
        throw new IllegalArgumentException("runtime golden case not found: " + name);
    }

    static List<TinygradRuntimeGoldenCase> parse(String json) {
        List<TinygradRuntimeGoldenCase> cases = new ArrayList<>();
        Matcher matcher = NAME_MARKER.matcher(json);
        while (matcher.find()) {
            int start = matcher.start();
            int end = findMatchingBrace(json, start);
            if (end < 0) {
                throw new IllegalArgumentException("unclosed case object at " + start);
            }
            cases.add(parseCase(json.substring(start, end + 1)));
        }
        return cases;
    }

    private static TinygradRuntimeGoldenCase parseCase(String block) {
        String name = stringField(block, "name");
        String path = stringField(block, "path");
        int[] logitsShape = intArrayField(block, "logitsShape");
        float[] logits = floatArrayField(block, "logits");
        float temperature = floatField(block, "temperature", 0f);
        int tensorSeed = intField(block, "tensorSeed", 0);
        int topK = intField(block, "topK", 0);
        float topP = floatField(block, "topP", 1f);
        float alphaF = floatField(block, "alphaFrequency", 0f);
        float alphaP = floatField(block, "alphaPresence", 0f);
        int[] alphaCounts = nullableIntArrayField(block, "alphaCounts");
        int runtimeToken = intField(block, "runtimeToken", -1);
        if (runtimeToken < 0) {
            throw new IllegalArgumentException("missing runtimeToken in " + name);
        }
        return new TinygradRuntimeGoldenCase(
                name, path, logitsShape, logits, temperature, tensorSeed,
                topK, topP, alphaF, alphaP, alphaCounts, runtimeToken);
    }

    private static int findMatchingBrace(String json, int start) {
        int depth = 0;
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    return i;
                }
            }
        }
        return -1;
    }

    private static String stringField(String block, String key) {
        Matcher m = Pattern.compile("\"" + key + "\":\\s*\"([^\"]*)\"").matcher(block);
        if (!m.find()) {
            throw new IllegalArgumentException("missing " + key);
        }
        return m.group(1);
    }

    private static int intField(String block, String key, int defaultValue) {
        Matcher m = Pattern.compile("\"" + key + "\":\\s*(-?\\d+)").matcher(block);
        if (!m.find()) {
            return defaultValue;
        }
        return Integer.parseInt(m.group(1));
    }

    private static float floatField(String block, String key, float defaultValue) {
        Matcher m = Pattern.compile("\"" + key + "\":\\s*(-?[\\d.]+(?:[Ee][+-]?\\d+)?)").matcher(block);
        if (!m.find()) {
            return defaultValue;
        }
        return Float.parseFloat(m.group(1));
    }

    private static float[] floatArrayField(String block, String key) {
        Matcher m = Pattern.compile("\"" + key + "\":\\s*\\[([^\\]]*)\\]", Pattern.DOTALL).matcher(block);
        if (!m.find()) {
            throw new IllegalArgumentException("missing array " + key);
        }
        String inner = m.group(1).replace('\n', ' ').trim();
        if (inner.isEmpty()) {
            return new float[0];
        }
        String[] parts = inner.split(",");
        float[] out = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Float.parseFloat(parts[i].trim());
        }
        return out;
    }

    private static int[] intArrayField(String block, String key) {
        Matcher m = Pattern.compile("\"" + key + "\":\\s*\\[([^\\]]*)\\]", Pattern.DOTALL).matcher(block);
        if (!m.find()) {
            throw new IllegalArgumentException("missing array " + key);
        }
        String inner = m.group(1).replace('\n', ' ').trim();
        if (inner.isEmpty()) {
            return new int[0];
        }
        String[] parts = inner.split(",");
        int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Integer.parseInt(parts[i].trim());
        }
        return out;
    }

    private static int[] nullableIntArrayField(String block, String key) {
        if (block.contains("\"" + key + "\": null")) {
            return null;
        }
        if (!block.contains("\"" + key + "\":")) {
            return null;
        }
        return intArrayField(block, key);
    }
}
