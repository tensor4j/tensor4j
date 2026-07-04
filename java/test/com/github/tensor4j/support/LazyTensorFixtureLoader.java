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

import com.github.tensor4j.core.lazy.LazyTensor;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Loads lazy-tensor parity fixtures ({@code tinygrad-lazy-tensor.json}). */
public final class LazyTensorFixtureLoader {

    private static final Pattern CASE_NAME = Pattern.compile("\"name\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern TENSOR = Pattern.compile(
            "\\{\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\"data\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern OP_PERMUTE = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"permute\"\\s*,\\s*\"order\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern OP_RESHAPE = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"reshape\"\\s*,\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern OP_EXPAND = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"expand\"\\s*,\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern OP_ADD = Pattern.compile("\\{\\s*\"op\"\\s*:\\s*\"add\"\\s*\\}");
    private static final Pattern OP_RELU = Pattern.compile("\\{\\s*\"op\"\\s*:\\s*\"relu\"\\s*\\}");
    private static final Pattern OP_MUL_SCALAR = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"mul_scalar\"\\s*,\\s*\"other\"\\s*:\\s*(\\{[^}]+\\})\\s*\\}");

    private LazyTensorFixtureLoader() {
    }

    public static List<LazyTensorCase> loadResource(String resourcePath) {
        try (var in = LazyTensorFixtureLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IllegalStateException("lazy tensor resource not found: " + resourcePath);
            }
            return parse(new String(in.readAllBytes()));
        } catch (Exception ex) {
            throw new IllegalStateException("failed to load lazy tensor fixtures: " + resourcePath, ex);
        }
    }

    static List<LazyTensorCase> parse(String json) {
        List<LazyTensorCase> cases = new ArrayList<>();
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

    public static LazyTensor buildLazyTensor(LazyTensorCase parityCase) {
        LazyTensor lazy = parityCase.hasBinaryInputs()
                ? null
                : LazyTensor.of(parityCase.input().data(), parityCase.input().shape());
        LazyTensor left = null;
        LazyTensor right = null;
        if (parityCase.hasBinaryInputs()) {
            left = LazyTensor.of(parityCase.inputs().get(0).data(), parityCase.inputs().get(0).shape());
            right = LazyTensor.of(parityCase.inputs().get(1).data(), parityCase.inputs().get(1).shape());
        }
        for (LazyTensorOp step : parityCase.ops()) {
            if (step.kind() == LazyTensorOpKind.ADD) {
                lazy = left.add(right);
            } else if (lazy == null) {
                throw new IllegalStateException("missing lazy root for case " + parityCase.name());
            } else if (step.kind() == LazyTensorOpKind.MUL_SCALAR) {
                LazyTensor other = LazyTensor.of(step.other().data(), step.other().shape());
                lazy = lazy.mul(other);
            } else {
                lazy = applyUnary(lazy, step);
            }
        }
        return lazy;
    }

    private static LazyTensor applyUnary(LazyTensor lazy, LazyTensorOp step) {
        return switch (step.kind()) {
            case RESHAPE -> lazy.reshape(step.intArg());
            case PERMUTE -> lazy.permute(step.intArg());
            case EXPAND -> lazy.expand(step.intArg());
            case RELU -> lazy.relu();
            default -> throw new IllegalArgumentException("unsupported unary op " + step.kind());
        };
    }

    private static LazyTensorCase parseCase(String body) {
        String name = matchGroup(CASE_NAME, body, 1);
        List<LazyTensorSpec> inputs = parseTensorList(extractArray(body, "\"inputs\""));
        LazyTensorSpec input = parseSingleTensor(extractObject(body, "\"input\""));
        LazyTensorSpec expected = parseSingleTensor(extractObject(body, "\"expected\""));
        List<LazyTensorOp> ops = parseOps(extractArray(body, "\"ops\""));
        return new LazyTensorCase(name, input, inputs, expected, ops);
    }

    private static List<LazyTensorOp> parseOps(String opsBody) {
        List<LazyTensorOp> ops = new ArrayList<>();
        if (opsBody == null || opsBody.isBlank()) {
            return ops;
        }
        int cursor = 0;
        while (cursor < opsBody.length()) {
            int objectStart = opsBody.indexOf('{', cursor);
            if (objectStart < 0) {
                break;
            }
            int objectEnd = findMatchingBracket(opsBody, objectStart);
            ops.add(parseSingleOp(opsBody.substring(objectStart, objectEnd + 1)));
            cursor = objectEnd + 1;
        }
        return ops;
    }

    private static LazyTensorOp parseSingleOp(String opBody) {
        Matcher permute = OP_PERMUTE.matcher(opBody);
        if (permute.find()) {
            return new LazyTensorOp(LazyTensorOpKind.PERMUTE, parseInts(permute.group(1)), null);
        }
        Matcher reshape = OP_RESHAPE.matcher(opBody);
        if (reshape.find()) {
            return new LazyTensorOp(LazyTensorOpKind.RESHAPE, parseInts(reshape.group(1)), null);
        }
        Matcher expand = OP_EXPAND.matcher(opBody);
        if (expand.find()) {
            return new LazyTensorOp(LazyTensorOpKind.EXPAND, parseInts(expand.group(1)), null);
        }
        if (OP_ADD.matcher(opBody).find()) {
            return new LazyTensorOp(LazyTensorOpKind.ADD, null, null);
        }
        if (OP_RELU.matcher(opBody).find()) {
            return new LazyTensorOp(LazyTensorOpKind.RELU, null, null);
        }
        Matcher mulScalar = OP_MUL_SCALAR.matcher(opBody);
        if (mulScalar.find()) {
            LazyTensorSpec other = parseSingleTensor(mulScalar.group(1));
            return new LazyTensorOp(LazyTensorOpKind.MUL_SCALAR, null, other);
        }
        throw new IllegalArgumentException("unsupported lazy tensor op: " + opBody);
    }

    private static LazyTensorSpec parseSingleTensor(String objectBody) {
        if (objectBody == null || objectBody.isBlank()) {
            return null;
        }
        Matcher matcher = TENSOR.matcher(objectBody);
        if (!matcher.find()) {
            return null;
        }
        return new LazyTensorSpec(parseInts(matcher.group(1)), parseFloats(matcher.group(2)));
    }

    private static List<LazyTensorSpec> parseTensorList(String arrayBody) {
        List<LazyTensorSpec> tensors = new ArrayList<>();
        if (arrayBody == null || arrayBody.isBlank()) {
            return tensors;
        }
        Matcher matcher = TENSOR.matcher(arrayBody);
        while (matcher.find()) {
            tensors.add(new LazyTensorSpec(parseInts(matcher.group(1)), parseFloats(matcher.group(2))));
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

    public enum LazyTensorOpKind {
        RESHAPE,
        PERMUTE,
        EXPAND,
        ADD,
        MUL_SCALAR,
        RELU
    }

    public record LazyTensorSpec(int[] shape, float[] data) {
    }

    public record LazyTensorOp(LazyTensorOpKind kind, int[] intArg, LazyTensorSpec other) {
    }

    public record LazyTensorCase(
            String name,
            LazyTensorSpec input,
            List<LazyTensorSpec> inputs,
            LazyTensorSpec expected,
            List<LazyTensorOp> ops) {

        public boolean hasBinaryInputs() {
            return inputs != null && inputs.size() >= 2;
        }
    }
}
