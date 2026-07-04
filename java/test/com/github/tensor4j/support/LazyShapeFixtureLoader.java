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

import com.github.tensor4j.core.lazy.LazyShape;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Loads tinygrad lazy-shape parity fixtures (shape-only, no buffer). */
public final class LazyShapeFixtureLoader {

    private static final Pattern CASE_NAME = Pattern.compile("\"name\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern LEAF_SHAPE = Pattern.compile("\"leaf\"\\s*:\\s*\\[([^\\]]*)\\]");
    private static final Pattern EXPECTED_SHAPE = Pattern.compile("\"expected_shape\"\\s*:\\s*\\[([^\\]]*)\\]");
    private static final Pattern OP_RESHAPE = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"reshape\"\\s*,\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern OP_PERMUTE = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"permute\"\\s*,\\s*\"order\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern OP_EXPAND = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"expand\"\\s*,\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");
    private static final Pattern OP_REDUCE = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"reduce\"\\s*,\\s*\"axis\"\\s*:\\s*(\\d+)\\s*\\}");
    private static final Pattern OP_BROADCAST = Pattern.compile(
            "\\{\\s*\"op\"\\s*:\\s*\"broadcast_with\"\\s*,\\s*\"other\"\\s*:\\s*\\[([^\\]]*)\\]\\s*\\}");

    private LazyShapeFixtureLoader() {
    }

    public static List<LazyShapeCase> loadResource(String resourcePath) {
        try (var in = LazyShapeFixtureLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IllegalStateException("lazy shape resource not found: " + resourcePath);
            }
            return parse(new String(in.readAllBytes()));
        } catch (Exception ex) {
            throw new IllegalStateException("failed to load lazy shape fixtures: " + resourcePath, ex);
        }
    }

    static List<LazyShapeCase> parse(String json) {
        List<LazyShapeCase> cases = new ArrayList<>();
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

    public static LazyShape buildLazyShape(LazyShapeCase parityCase) {
        LazyShape lazy = LazyShape.leaf(parityCase.leafShape());
        for (LazyShapeOp step : parityCase.ops()) {
            lazy = applyOp(lazy, step);
        }
        return lazy;
    }

    private static LazyShape applyOp(LazyShape lazy, LazyShapeOp step) {
        return switch (step.kind()) {
            case RESHAPE -> lazy.reshape(step.intArg());
            case PERMUTE -> lazy.permute(step.intArg());
            case EXPAND -> lazy.expand(step.intArg());
            case REDUCE -> lazy.reduceAxis(step.intArg()[0]);
            case BROADCAST_WITH -> LazyShape.broadcast(lazy, LazyShape.leaf(step.intArg()));
        };
    }

    private static LazyShapeCase parseCase(String body) {
        String name = matchGroup(CASE_NAME, body, 1);
        int[] leaf = parseInts(matchGroup(LEAF_SHAPE, body, 1));
        int[] expected = parseInts(matchGroup(EXPECTED_SHAPE, body, 1));
        List<LazyShapeOp> ops = parseOps(extractArray(body, "\"ops\""));
        return new LazyShapeCase(name, leaf, ops, expected);
    }

    private static List<LazyShapeOp> parseOps(String opsBody) {
        return parseOpsInDocumentOrder(opsBody);
    }

    private static List<LazyShapeOp> parseOpsInDocumentOrder(String opsBody) {
        List<LazyShapeOp> ops = new ArrayList<>();
        int cursor = 0;
        while (cursor < opsBody.length()) {
            int objectStart = opsBody.indexOf('{', cursor);
            if (objectStart < 0) {
                break;
            }
            int objectEnd = findMatchingBracket(opsBody, objectStart);
            String opBody = opsBody.substring(objectStart, objectEnd + 1);
            ops.add(parseSingleOp(opBody));
            cursor = objectEnd + 1;
        }
        return ops;
    }

    private static LazyShapeOp parseSingleOp(String opBody) {
        Matcher reshape = OP_RESHAPE.matcher(opBody);
        if (reshape.find()) {
            return new LazyShapeOp(LazyShapeOpKind.RESHAPE, parseInts(reshape.group(1)));
        }
        Matcher permute = OP_PERMUTE.matcher(opBody);
        if (permute.find()) {
            return new LazyShapeOp(LazyShapeOpKind.PERMUTE, parseInts(permute.group(1)));
        }
        Matcher expand = OP_EXPAND.matcher(opBody);
        if (expand.find()) {
            return new LazyShapeOp(LazyShapeOpKind.EXPAND, parseInts(expand.group(1)));
        }
        Matcher reduce = OP_REDUCE.matcher(opBody);
        if (reduce.find()) {
            return new LazyShapeOp(LazyShapeOpKind.REDUCE, new int[] {Integer.parseInt(reduce.group(1))});
        }
        Matcher broadcast = OP_BROADCAST.matcher(opBody);
        if (broadcast.find()) {
            return new LazyShapeOp(LazyShapeOpKind.BROADCAST_WITH, parseInts(broadcast.group(1)));
        }
        throw new IllegalArgumentException("unsupported lazy shape op: " + opBody);
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

    public enum LazyShapeOpKind {
        RESHAPE,
        PERMUTE,
        EXPAND,
        REDUCE,
        BROADCAST_WITH
    }

    public record LazyShapeOp(LazyShapeOpKind kind, int[] intArg) {
    }

    public record LazyShapeCase(String name, int[] leafShape, List<LazyShapeOp> ops, int[] expectedShape) {
    }
}
