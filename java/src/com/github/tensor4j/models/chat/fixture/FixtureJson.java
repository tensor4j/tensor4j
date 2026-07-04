/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.fixture;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Minimal JSON field parsing for checked-in fixture files (no third-party JSON). */
final class FixtureJson {

    private FixtureJson() {
    }

    static String stringField(String json, String key) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\":\\s*\"((?:\\\\.|[^\"\\\\])*)\"").matcher(json);
        if (!m.find()) {
            throw new IllegalArgumentException("missing string field " + key);
        }
        return unescape(m.group(1));
    }

    static boolean booleanField(String json, String key, boolean defaultValue) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\":\\s*(true|false)").matcher(json);
        if (!m.find()) {
            return defaultValue;
        }
        return Boolean.parseBoolean(m.group(1));
    }

    static int intField(String json, String key, int defaultValue) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\":\\s*(-?\\d+)").matcher(json);
        if (!m.find()) {
            return defaultValue;
        }
        return Integer.parseInt(m.group(1));
    }

    static String[] stringArrayField(String json, String key) {
        int start = arrayStart(json, key);
        int end = matchingBracket(json, start, '[', ']');
        return parseStringElements(json, start + 1, end);
    }

    static int[] intArrayField(String json, String key) {
        int start = arrayStart(json, key);
        int end = matchingBracket(json, start, '[', ']');
        String inner = json.substring(start + 1, end).trim();
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

    static String readResource(String path) {
        try (InputStream in = ChatDemoVocab.class.getResourceAsStream(path)) {
            if (in == null) {
                throw new IllegalStateException("missing resource " + path);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new IllegalStateException("failed to read resource " + path, e);
        }
    }

    private static int arrayStart(String json, String key) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\":\\s*\\[").matcher(json);
        if (!m.find()) {
            throw new IllegalArgumentException("missing array field " + key);
        }
        return m.end() - 1;
    }

    private static int matchingBracket(String json, int openIdx, char open, char close) {
        int depth = 0;
        boolean inString = false;
        boolean escape = false;
        for (int i = openIdx; i < json.length(); i++) {
            char c = json.charAt(i);
            if (inString) {
                if (escape) {
                    escape = false;
                } else if (c == '\\') {
                    escape = true;
                } else if (c == '"') {
                    inString = false;
                }
                continue;
            }
            if (c == '"') {
                inString = true;
            } else if (c == open) {
                depth++;
            } else if (c == close) {
                depth--;
                if (depth == 0) {
                    return i;
                }
            }
        }
        throw new IllegalArgumentException("unclosed bracket at " + openIdx);
    }

    private static String[] parseStringElements(String json, int start, int end) {
        List<String> out = new ArrayList<>();
        int i = start;
        while (i < end) {
            while (i < end && Character.isWhitespace(json.charAt(i))) {
                i++;
            }
            if (i >= end) {
                break;
            }
            if (json.charAt(i) == ',') {
                i++;
                continue;
            }
            if (json.charAt(i) != '"') {
                throw new IllegalArgumentException("expected string at " + i);
            }
            i++;
            StringBuilder sb = new StringBuilder();
            boolean escape = false;
            while (i < end) {
                char c = json.charAt(i++);
                if (escape) {
                    switch (c) {
                        case '"', '\\', '/' -> sb.append(c);
                        case 'b' -> sb.append('\b');
                        case 'f' -> sb.append('\f');
                        case 'n' -> sb.append('\n');
                        case 'r' -> sb.append('\r');
                        case 't' -> sb.append('\t');
                        case 'u' -> {
                            if (i + 4 > end) {
                                throw new IllegalArgumentException("bad unicode escape");
                            }
                            int code = Integer.parseInt(json, i, i + 4, 16);
                            sb.appendCodePoint(code);
                            i += 4;
                        }
                        default -> throw new IllegalArgumentException("bad escape \\" + c);
                    }
                    escape = false;
                } else if (c == '\\') {
                    escape = true;
                } else if (c == '"') {
                    break;
                } else {
                    sb.append(c);
                }
            }
            out.add(sb.toString());
        }
        return out.toArray(new String[0]);
    }

    private static String unescape(String value) {
        StringBuilder sb = new StringBuilder();
        boolean escape = false;
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (escape) {
                switch (c) {
                    case '"', '\\', '/' -> sb.append(c);
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        if (i + 4 >= value.length()) {
                            throw new IllegalArgumentException("bad unicode escape");
                        }
                        int code = Integer.parseInt(value, i + 1, i + 5, 16);
                        sb.appendCodePoint(code);
                        i += 4;
                    }
                    default -> throw new IllegalArgumentException("bad escape \\" + c);
                }
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
