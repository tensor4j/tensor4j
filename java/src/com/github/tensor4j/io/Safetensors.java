/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.io;

import com.github.tensor4j.core.Tensor;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Safetensors reader/writer ({@code tinygrad.nn.state.safe_load} / {@code safe_save}).
 * Supports F32 tensors only — sufficient for algebra MLP and tinygrad parity exports.
 */
public final class Safetensors {

    private static final Pattern TENSOR_HEADER = Pattern.compile(
            "\"([^\"]+)\"\\s*:\\s*\\{\\s*\"dtype\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]"
                    + "\\s*,\\s*\"data_offsets\"\\s*:\\s*\\[(\\d+)\\s*,\\s*(\\d+)\\]\\s*\\}");

    private Safetensors() {
    }

    public static Map<String, Tensor> load(Path path) throws IOException {
        return load(Files.readAllBytes(path));
    }

    public static Map<String, Tensor> loadResource(String resourcePath) throws IOException {
        try (InputStream in = Safetensors.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IOException("resource not found: " + resourcePath);
            }
            return load(in.readAllBytes());
        }
    }

    public static Map<String, Tensor> load(byte[] fileBytes) {
        if (fileBytes.length < 8) {
            throw new IllegalArgumentException("safetensors file too small");
        }
        ByteBuffer headerLenBuf = ByteBuffer.wrap(fileBytes, 0, 8).order(ByteOrder.LITTLE_ENDIAN);
        long headerLen = headerLenBuf.getLong();
        if (headerLen < 0 || headerLen > fileBytes.length - 8) {
            throw new IllegalArgumentException("invalid safetensors header length " + headerLen);
        }
        int headerEnd = 8 + (int) headerLen;
        String headerJson = new String(fileBytes, 8, (int) headerLen, StandardCharsets.UTF_8);

        Map<String, Tensor> tensors = new LinkedHashMap<>();
        Matcher matcher = TENSOR_HEADER.matcher(headerJson);
        while (matcher.find()) {
            String name = matcher.group(1);
            if ("__metadata__".equals(name)) {
                continue;
            }
            String dtype = matcher.group(2);
            if (!"F32".equals(dtype)) {
                throw new IllegalArgumentException("unsupported safetensors dtype " + dtype + " for " + name);
            }
            int[] shape = parseInts(matcher.group(3));
            int start = Integer.parseInt(matcher.group(4));
            int end = Integer.parseInt(matcher.group(5));
            int dataStart = headerEnd + start;
            int dataEnd = headerEnd + end;
            float[] data = readF32Slice(fileBytes, dataStart, dataEnd);
            validateElementCount(name, shape, data.length);
            tensors.put(name, Tensor.of(data, shape));
        }
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("no tensors found in safetensors header");
        }
        return tensors;
    }

    public static void save(Path path, Map<String, Tensor> tensors) throws IOException {
        Files.createDirectories(path.getParent() == null ? Path.of(".") : path.getParent());
        Files.write(path, encode(tensors));
    }

    public static byte[] encode(Map<String, Tensor> tensors) {
        int offset = 0;
        StringBuilder header = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Tensor> entry : tensors.entrySet()) {
            Tensor tensor = entry.getValue();
            int nbytes = tensor.data().length * Float.BYTES;
            if (!first) {
                header.append(',');
            }
            first = false;
            header.append('"').append(escapeJson(entry.getKey())).append("\":{");
            header.append("\"dtype\":\"F32\",");
            header.append("\"shape\":").append(shapeJson(tensor.shape().dims())).append(',');
            header.append("\"data_offsets\":[").append(offset).append(',').append(offset + nbytes).append("]}");
            offset += nbytes;
        }
        header.append('}');

        byte[] headerBytes = padHeader(header.toString());
        ByteBuffer out = ByteBuffer.allocate(8 + headerBytes.length + offset).order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(headerBytes.length);
        out.put(headerBytes);
        for (Map.Entry<String, Tensor> entry : tensors.entrySet()) {
            for (float value : entry.getValue().data()) {
                out.putFloat(value);
            }
        }
        return out.array();
    }

    private static byte[] padHeader(String json) {
        byte[] raw = json.getBytes(StandardCharsets.UTF_8);
        int paddedLen = roundUp(raw.length, 8);
        byte[] header = new byte[paddedLen];
        System.arraycopy(raw, 0, header, 0, raw.length);
        for (int i = raw.length; i < paddedLen; i++) {
            header[i] = 0x20;
        }
        return header;
    }

    private static int roundUp(int value, int align) {
        return ((value + align - 1) / align) * align;
    }

    private static String shapeJson(int[] dims) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(dims[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    private static String escapeJson(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static float[] readF32Slice(byte[] fileBytes, int start, int end) {
        if (start < 0 || end < start || end > fileBytes.length) {
            throw new IllegalArgumentException("invalid safetensors data slice");
        }
        int count = (end - start) / Float.BYTES;
        float[] out = new float[count];
        ByteBuffer buffer = ByteBuffer.wrap(fileBytes, start, end - start).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            out[i] = buffer.getFloat();
        }
        return out;
    }

    private static void validateElementCount(String name, int[] shape, int count) {
        int expected = 1;
        for (int dim : shape) {
            expected *= dim;
        }
        if (expected != count) {
            throw new IllegalArgumentException(
                    "shape/data mismatch for " + name + ": expected " + expected + " floats, got " + count);
        }
    }

    private static int[] parseInts(String raw) {
        if (raw.isBlank()) {
            return new int[0];
        }
        String[] parts = raw.split(",");
        int[] values = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Integer.parseInt(parts[i].trim());
        }
        return values;
    }
}
