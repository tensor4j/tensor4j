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

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/** GPT-2 byte-to-unicode mapping (llama.cpp BPE byte_encode path). */
final class Gpt2ByteEncoder {

    private static final Map<Integer, String> BYTE_TO_CHAR = buildByteToChar();
    private static final Map<String, Integer> CHAR_TO_BYTE = buildCharToByte();

    private Gpt2ByteEncoder() {
    }

    static String encode(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        StringBuilder out = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            out.append(BYTE_TO_CHAR.get(b & 0xFF));
        }
        return out.toString();
    }

    static String decode(String text) {
        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++) {
            Integer b = CHAR_TO_BYTE.get(text.substring(i, i + 1));
            if (b == null) {
                throw new IllegalArgumentException("unknown byte token char at " + i);
            }
            bytes[i] = b.byteValue();
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static Map<Integer, String> buildByteToChar() {
        Map<Integer, String> map = new HashMap<>();
        int n = 0;
        for (int b = 33; b <= 126; b++) {
            map.put(b, String.valueOf((char) b));
            n++;
        }
        for (int b = 161; b <= 172; b++) {
            map.put(b, String.valueOf((char) b));
            n++;
        }
        for (int b = 174; b <= 255; b++) {
            map.put(b, String.valueOf((char) b));
            n++;
        }
        for (int b = 0; b <= 255; b++) {
            if (!map.containsKey(b)) {
                map.put(b, String.valueOf((char) (256 + n)));
                n++;
            }
        }
        return map;
    }

    private static Map<String, Integer> buildCharToByte() {
        Map<String, Integer> map = new HashMap<>();
        for (Map.Entry<Integer, String> entry : BYTE_TO_CHAR.entrySet()) {
            map.put(entry.getValue(), entry.getKey());
        }
        return map;
    }
}
