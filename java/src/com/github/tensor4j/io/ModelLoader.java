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

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.nn.Linear;
import com.github.tensor4j.nn.Sequential;

/**
 * Loads tinygrad-compatible tensor4j model bundles ({@code .t4j.json} or {@code .safetensors}).
 * Tensor names follow tinygrad state-dict keys, e.g. {@code fc1.weight}.
 */
public final class ModelLoader {

    private static final Pattern TENSOR_BLOCK = Pattern.compile(
            "\"([^\"]+)\"\\s*:\\s*\\{\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\"data\"\\s*:\\s*\\[([^\\]]*)\\]",
            Pattern.DOTALL);

    private ModelLoader() {
    }

    public static Map<String, Tensor> load(Path path) throws IOException {
        return load(path, WeightFormat.fromPath(path));
    }

    public static Map<String, Tensor> load(Path path, WeightFormat format) throws IOException {
        if (format == WeightFormat.SAFETENSORS) {
            return Safetensors.load(path);
        }
        String json = Files.readString(path, StandardCharsets.UTF_8);
        return parse(json);
    }

    public static Map<String, Tensor> loadResource(String resourcePath) throws IOException {
        return loadResource(resourcePath, WeightFormat.fromPath(Path.of(resourcePath)));
    }

    public static Map<String, Tensor> loadResource(String resourcePath, WeightFormat format) throws IOException {
        if (format == WeightFormat.SAFETENSORS) {
            return Safetensors.loadResource(resourcePath);
        }
        try (InputStream in = ModelLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IOException("resource not found: " + resourcePath);
            }
            return parse(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    public static void save(Path path, Map<String, Tensor> tensors) throws IOException {
        save(path, tensors, WeightFormat.fromPath(path));
    }

    public static void save(Path path, Map<String, Tensor> tensors, WeightFormat format) throws IOException {
        if (format == WeightFormat.SAFETENSORS) {
            Safetensors.save(path, tensors);
            return;
        }
        Files.createDirectories(path.getParent() == null ? Path.of(".") : path.getParent());
        Files.writeString(path, export(tensors));
    }

    public static Map<String, Tensor> exportTensors(Sequential network) {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        for (Sequential.Layer layer : network.layers()) {
            if (layer.module() instanceof Linear linear) {
                String prefix = layer.name();
                tensors.put(prefix + ".weight", linear.weight());
                tensors.put(prefix + ".bias", linear.bias());
            }
        }
        return tensors;
    }

    public static void applyToSequential(Sequential network, Map<String, Tensor> tensors) {
        for (Sequential.Layer layer : network.layers()) {
            if (layer.module() instanceof Linear linear) {
                String prefix = layer.name();
                Tensor weight = tensors.get(prefix + ".weight");
                Tensor bias = tensors.get(prefix + ".bias");
                if (weight != null) {
                    System.arraycopy(weight.data(), 0, linear.weight().data(), 0, weight.data().length);
                }
                if (bias != null) {
                    System.arraycopy(bias.data(), 0, linear.bias().data(), 0, bias.data().length);
                }
            }
        }
    }

    public static String export(Sequential network) {
        return export(exportTensors(network));
    }

    public static String export(Map<String, Tensor> tensors) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n  \"format\": \"tensor4j-v1\",\n");
        sb.append("  \"framework\": \"tinygrad-compatible\",\n");
        sb.append("  \"dtype\": \"float32\",\n  \"tensors\": {\n");
        boolean first = true;
        for (Map.Entry<String, Tensor> entry : tensors.entrySet()) {
            if (!first) {
                sb.append(",\n");
            }
            first = false;
            appendTensor(sb, entry.getKey(), entry.getValue());
        }
        sb.append("\n  }\n}\n");
        return sb.toString();
    }

    private static void appendTensor(StringBuilder sb, String name, Tensor tensor) {
        sb.append("    \"").append(name).append("\": { \"shape\": [");
        int[] dims = tensor.shape().dims();
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(dims[i]);
        }
        sb.append("], \"data\": [");
        float[] data = tensor.data();
        for (int i = 0; i < data.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(trimFloat(data[i]));
        }
        sb.append("] }");
    }

    private static String trimFloat(float value) {
        if (value == (long) value) {
            return Long.toString((long) value);
        }
        return Float.toString(value);
    }

    static Map<String, Tensor> parse(String json) {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        Matcher matcher = TENSOR_BLOCK.matcher(json);
        while (matcher.find()) {
            String name = matcher.group(1);
            int[] shape = parseInts(matcher.group(2));
            float[] data = parseFloats(matcher.group(3));
            tensors.put(name, Tensor.of(data, shape));
        }
        return tensors;
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

    private static float[] parseFloats(String raw) {
        if (raw.isBlank()) {
            return new float[0];
        }
        String[] parts = raw.split(",");
        float[] values = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }
}
