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

/** Loads tinygrad-exported parity fixtures from classpath JSON. */
public final class ParityFixtureLoader {

    private static final Pattern CASE_NAME = Pattern.compile("\"name\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern CASE_OP = Pattern.compile("\"op\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern TENSOR = Pattern.compile(
            "\\{\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\"data\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern ARGS_SHAPE = Pattern.compile(
            "\"args\"\\s*:\\s*\\{[^}]*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]");
    private static final Pattern ARGS_ORDER = Pattern.compile(
            "\"args\"\\s*:\\s*\\{[^}]*\"order\"\\s*:\\s*\\[([^\\]]*)\\]");
    private static final Pattern AXIS = Pattern.compile("\"axis\"\\s*:\\s*(\\d+)");

    private ParityFixtureLoader() {
    }

    public static List<ParityCase> loadResource(String resourcePath) {
        try (var in = ParityFixtureLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IllegalStateException("parity resource not found: " + resourcePath);
            }
            return parse(new String(in.readAllBytes()));
        } catch (Exception ex) {
            throw new IllegalStateException("failed to load parity fixtures: " + resourcePath, ex);
        }
    }

    static List<ParityCase> parse(String json) {
        List<ParityCase> cases = new ArrayList<>();
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

    private static ParityCase parseCase(String body) {
        String name = matchGroup(CASE_NAME, body, 1);
        String op = matchGroup(CASE_OP, body, 1);
        List<ParityTensor> inputs = parseTensorList(extractArray(body, "\"inputs\""));
        ParityTensor input = parseSingleTensor(extractObject(body, "\"input\""));
        ParityTensor expected = parseSingleTensor(extractObject(body, "\"expected\""));
        int[] argShape = null;
        int[] argOrder = null;
        Matcher shapeArgs = ARGS_SHAPE.matcher(body);
        if (shapeArgs.find()) {
            argShape = parseInts(shapeArgs.group(1));
        }
        Matcher orderArgs = ARGS_ORDER.matcher(body);
        if (orderArgs.find()) {
            argOrder = parseInts(orderArgs.group(1));
        }
        int argAxis = -1;
        Matcher axisMatcher = AXIS.matcher(body);
        if (axisMatcher.find()) {
            argAxis = Integer.parseInt(axisMatcher.group(1));
        }
        return new ParityCase(name, op, input, inputs, expected, argShape, argOrder, argAxis);
    }

    private static ParityTensor parseSingleTensor(String objectBody) {
        if (objectBody == null || objectBody.isBlank()) {
            return null;
        }
        Matcher matcher = TENSOR.matcher(objectBody);
        if (!matcher.find()) {
            return null;
        }
        return new ParityTensor(parseInts(matcher.group(1)), parseFloats(matcher.group(2)));
    }

    private static List<ParityTensor> parseTensorList(String arrayBody) {
        List<ParityTensor> tensors = new ArrayList<>();
        if (arrayBody == null || arrayBody.isBlank()) {
            return tensors;
        }
        Matcher matcher = TENSOR.matcher(arrayBody);
        while (matcher.find()) {
            tensors.add(new ParityTensor(parseInts(matcher.group(1)), parseFloats(matcher.group(2))));
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

    public record ParityTensor(int[] shape, float[] data) {
    }

    public record ParityCase(
            String name,
            String op,
            ParityTensor input,
            List<ParityTensor> inputs,
            ParityTensor expected,
            int[] argShape,
            int[] argOrder,
            int argAxis) {
    }
}
