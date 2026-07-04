/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.support;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Loads {@code java/resources/parity/tinygrad-ggml-ops.json} for infer-kernel parity tests. */
public final class GgmlParityFixtureLoader {

    private static final Pattern CASE_NAME = Pattern.compile("\"name\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern CASE_OP = Pattern.compile("\"op\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern TOLERANCE = Pattern.compile("\"tolerance\"\\s*:\\s*([0-9.eE+-]+)");
    private static final Pattern TENSOR = Pattern.compile(
            "\\{\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\"data\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern FLOAT_ARG = Pattern.compile(
            "\"(eps|scale|freq_base|freq_scale|yarn_ext_factor|yarn_attn_factor|yarn_beta_fast|yarn_beta_slow)\""
                    + "\\s*:\\s*([0-9.eE+-]+)");
    private static final Pattern INT_ARG = Pattern.compile(
            "\"(n_heads|head_dim|rope_dim|past_kv|yarn_orig_ctx)\"\\s*:\\s*(\\d+)");
    private static final Pattern SCALING = Pattern.compile("\"scaling\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern INT_ARRAY = Pattern.compile("\"positions\"\\s*:\\s*\\[([^\\]]*)\\]");

    private GgmlParityFixtureLoader() {
    }

    public static List<GgmlParityCase> loadResource(String resourcePath) {
        try (var in = GgmlParityFixtureLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IllegalStateException("ggml parity resource not found: " + resourcePath);
            }
            return parse(new String(in.readAllBytes()));
        } catch (Exception ex) {
            throw new IllegalStateException("failed to load ggml parity fixtures: " + resourcePath, ex);
        }
    }

    static List<GgmlParityCase> parse(String json) {
        List<GgmlParityCase> cases = new ArrayList<>();
        int casesIndex = json.indexOf("\"cases\"");
        if (casesIndex < 0) {
            return cases;
        }
        int arrayStart = json.indexOf('[', casesIndex);
        int arrayEnd = findMatchingBracket(json, arrayStart);
        String casesBody = json.substring(arrayStart + 1, arrayEnd);
        int cursor = 0;
        while (cursor < casesBody.length()) {
            int objectStart = casesBody.indexOf('{', cursor);
            if (objectStart < 0) {
                break;
            }
            int objectEnd = findMatchingBracket(casesBody, objectStart);
            cases.add(parseCase(casesBody.substring(objectStart, objectEnd + 1)));
            cursor = objectEnd + 1;
        }
        return cases;
    }

    private static GgmlParityCase parseCase(String body) {
        String name = matchGroup(CASE_NAME, body, 1);
        String op = matchGroup(CASE_OP, body, 1);
        float tolerance = 1e-5f;
        Matcher tol = TOLERANCE.matcher(body);
        if (tol.find()) {
            tolerance = Float.parseFloat(tol.group(1));
        }
        List<ParityFixtureLoader.ParityTensor> inputs = parseTensorList(extractArray(body, "\"inputs\""));
        ParityFixtureLoader.ParityTensor input = parseSingleTensor(extractObject(body, "\"input\""));
        ParityFixtureLoader.ParityTensor expected = parseSingleTensor(extractObject(body, "\"expected\""));
        float eps = 1e-5f;
        float scale = 1f;
        float freqBase = 10000f;
        float freqScale = 1f;
        float yarnExtFactor = 0f;
        float yarnAttnFactor = 1f;
        float yarnBetaFast = 32f;
        float yarnBetaSlow = 1f;
        int nHeads = 1;
        int headDim = 4;
        int ropeDim = 4;
        int pastKv = 0;
        int yarnOrigCtx = 0;
        String scaling = "none";
        int[] positions = new int[] {0};
        Matcher floatArg = FLOAT_ARG.matcher(body);
        while (floatArg.find()) {
            switch (floatArg.group(1)) {
                case "eps" -> eps = Float.parseFloat(floatArg.group(2));
                case "scale" -> scale = Float.parseFloat(floatArg.group(2));
                case "freq_base" -> freqBase = Float.parseFloat(floatArg.group(2));
                case "freq_scale" -> freqScale = Float.parseFloat(floatArg.group(2));
                case "yarn_ext_factor" -> yarnExtFactor = Float.parseFloat(floatArg.group(2));
                case "yarn_attn_factor" -> yarnAttnFactor = Float.parseFloat(floatArg.group(2));
                case "yarn_beta_fast" -> yarnBetaFast = Float.parseFloat(floatArg.group(2));
                case "yarn_beta_slow" -> yarnBetaSlow = Float.parseFloat(floatArg.group(2));
                default -> {
                }
            }
        }
        Matcher intArg = INT_ARG.matcher(body);
        while (intArg.find()) {
            switch (intArg.group(1)) {
                case "n_heads" -> nHeads = Integer.parseInt(intArg.group(2));
                case "head_dim" -> headDim = Integer.parseInt(intArg.group(2));
                case "rope_dim" -> ropeDim = Integer.parseInt(intArg.group(2));
                case "past_kv" -> pastKv = Integer.parseInt(intArg.group(2));
                case "yarn_orig_ctx" -> yarnOrigCtx = Integer.parseInt(intArg.group(2));
                default -> {
                }
            }
        }
        Matcher scalingMatch = SCALING.matcher(body);
        if (scalingMatch.find()) {
            scaling = scalingMatch.group(1);
        }
        Matcher pos = INT_ARRAY.matcher(body);
        if (pos.find()) {
            positions = parseInts(pos.group(1));
        }
        return new GgmlParityCase(
                name, op, tolerance, input, inputs, expected, eps, scale, freqBase, freqScale, scaling,
                yarnExtFactor, yarnAttnFactor, yarnBetaFast, yarnBetaSlow, yarnOrigCtx, nHeads, headDim, ropeDim,
                pastKv, positions);
    }

    private static ParityFixtureLoader.ParityTensor parseSingleTensor(String objectBody) {
        if (objectBody == null || objectBody.isBlank()) {
            return null;
        }
        Matcher matcher = TENSOR.matcher(objectBody);
        if (!matcher.find()) {
            return null;
        }
        return new ParityFixtureLoader.ParityTensor(parseInts(matcher.group(1)), parseFloats(matcher.group(2)));
    }

    private static List<ParityFixtureLoader.ParityTensor> parseTensorList(String arrayBody) {
        List<ParityFixtureLoader.ParityTensor> tensors = new ArrayList<>();
        if (arrayBody == null || arrayBody.isBlank()) {
            return tensors;
        }
        Matcher matcher = TENSOR.matcher(arrayBody);
        while (matcher.find()) {
            tensors.add(new ParityFixtureLoader.ParityTensor(parseInts(matcher.group(1)), parseFloats(matcher.group(2))));
        }
        return tensors;
    }

    private static String extractArray(String body, String key) {
        int keyIndex = body.indexOf(key);
        if (keyIndex < 0) {
            return null;
        }
        int start = body.indexOf('[', keyIndex);
        if (start < 0) {
            return null;
        }
        int end = findMatchingBracket(body, start);
        return body.substring(start + 1, end);
    }

    private static String extractObject(String body, String key) {
        int keyIndex = body.indexOf(key);
        if (keyIndex < 0) {
            return null;
        }
        int start = body.indexOf('{', keyIndex);
        if (start < 0) {
            return null;
        }
        int end = findMatchingBracket(body, start);
        return body.substring(start, end + 1);
    }

    private static int findMatchingBracket(String text, int openIndex) {
        char open = text.charAt(openIndex);
        char close = open == '{' ? '}' : ']';
        int depth = 0;
        for (int i = openIndex; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ch == open) {
                depth++;
            } else if (ch == close) {
                depth--;
                if (depth == 0) {
                    return i;
                }
            }
        }
        throw new IllegalArgumentException("unbalanced brackets at " + openIndex);
    }

    private static String matchGroup(Pattern pattern, String body, int group) {
        Matcher matcher = pattern.matcher(body);
        if (!matcher.find()) {
            throw new IllegalArgumentException("missing pattern " + pattern + " in case body");
        }
        return matcher.group(group);
    }

    private static int[] parseInts(String raw) {
        if (raw == null || raw.isBlank()) {
            return new int[0];
        }
        String[] parts = raw.split(",");
        int[] values = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Integer.parseInt(parts[i].trim());
        }
        return values;
    }

    private static float[] parseFloats(String raw) {
        if (raw == null || raw.isBlank()) {
            return new float[0];
        }
        String[] parts = raw.split(",");
        float[] values = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }

    public record GgmlParityCase(
            String name,
            String op,
            float tolerance,
            ParityFixtureLoader.ParityTensor input,
            List<ParityFixtureLoader.ParityTensor> inputs,
            ParityFixtureLoader.ParityTensor expected,
            float eps,
            float scale,
            float freqBase,
            float freqScale,
            String scaling,
            float yarnExtFactor,
            float yarnAttnFactor,
            float yarnBetaFast,
            float yarnBetaSlow,
            int yarnOrigCtx,
            int nHeads,
            int headDim,
            int ropeDim,
            int pastKv,
            int[] positions) {
    }
}
