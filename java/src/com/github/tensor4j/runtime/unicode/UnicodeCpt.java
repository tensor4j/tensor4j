/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.unicode;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** UTF-8 codepoint helpers ({@code unicode_cpts_from_utf8} in llama.cpp). */
public final class UnicodeCpt {

    private static final int REPLACEMENT = 0xFFFD;

    private UnicodeCpt() {
    }

    public static int[] codepointsFromUtf8(String text) {
        List<Integer> out = new ArrayList<>(text.length());
        int offset = 0;
        while (offset < text.length()) {
            try {
                int[] read = readCodepoint(text, offset);
                out.add(read[0]);
                offset = read[1];
            } catch (IllegalArgumentException ex) {
                offset++;
                out.add(REPLACEMENT);
            }
        }
        int[] cpts = new int[out.size()];
        for (int i = 0; i < out.size(); i++) {
            cpts[i] = out.get(i);
        }
        return cpts;
    }

    public static String codepointToUtf8(int cpt) {
        if (cpt <= 0x7F) {
            return String.valueOf((char) cpt);
        }
        if (cpt <= 0x7FF) {
            byte[] bytes = new byte[] {
                (byte) (0xC0 | ((cpt >> 6) & 0x1F)),
                (byte) (0x80 | (cpt & 0x3F)),
            };
            return new String(bytes, StandardCharsets.UTF_8);
        }
        if (cpt <= 0xFFFF) {
            byte[] bytes = new byte[] {
                (byte) (0xE0 | ((cpt >> 12) & 0x0F)),
                (byte) (0x80 | ((cpt >> 6) & 0x3F)),
                (byte) (0x80 | (cpt & 0x3F)),
            };
            return new String(bytes, StandardCharsets.UTF_8);
        }
        if (cpt <= 0x10FFFF) {
            byte[] bytes = new byte[] {
                (byte) (0xF0 | ((cpt >> 18) & 0x07)),
                (byte) (0x80 | ((cpt >> 12) & 0x3F)),
                (byte) (0x80 | ((cpt >> 6) & 0x3F)),
                (byte) (0x80 | (cpt & 0x3F)),
            };
            return new String(bytes, StandardCharsets.UTF_8);
        }
        throw new IllegalArgumentException("invalid codepoint " + cpt);
    }

    public static String sliceToUtf8(int[] cpts, int start, int end) {
        StringBuilder out = new StringBuilder();
        for (int i = start; i < end; i++) {
            out.append(codepointToUtf8(cpts[i]));
        }
        return out.toString();
    }

    public static int toLower(int cpt) {
        return Character.toLowerCase(cpt);
    }

    private static int[] readCodepoint(String text, int offset) {
        char first = text.charAt(offset);
        if (Character.isHighSurrogate(first)) {
            if (offset + 1 >= text.length()) {
                throw new IllegalArgumentException("truncated surrogate");
            }
            char second = text.charAt(offset + 1);
            if (!Character.isLowSurrogate(second)) {
                throw new IllegalArgumentException("invalid surrogate pair");
            }
            return new int[] {Character.toCodePoint(first, second), offset + 2};
        }
        return new int[] {first, offset + 1};
    }
}
