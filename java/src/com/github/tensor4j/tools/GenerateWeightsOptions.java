/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.tools;

import java.nio.file.Path;
import java.util.Locale;
import com.github.tensor4j.io.WeightFormat;

/** Parsed CLI options for {@link GenerateWeights}. */
public final class GenerateWeightsOptions {

    private final WeightProfile profile;
    private final Path output;
    private final WeightFormat format;
    private final int[] layerSizes;
    private final int epochs;
    private final float learningRate;
    private final int batchSize;
    private final boolean help;

    GenerateWeightsOptions(
            WeightProfile profile,
            Path output,
            WeightFormat format,
            int[] layerSizes,
            int epochs,
            float learningRate,
            int batchSize,
            boolean help) {
        this.profile = profile;
        this.output = output;
        this.format = format;
        this.layerSizes = layerSizes;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.help = help;
    }

    public WeightProfile profile() {
        return profile;
    }

    public Path output() {
        return output;
    }

    public WeightFormat format() {
        return format;
    }

    public int[] layerSizes() {
        return layerSizes;
    }

    public int epochs() {
        return epochs;
    }

    public float learningRate() {
        return learningRate;
    }

    public int batchSize() {
        return batchSize;
    }

    public boolean help() {
        return help;
    }

    static GenerateWeightsOptions parse(String[] args) {
        WeightProfile profile = WeightProfile.GENERAL;
        Path output = null;
        WeightFormat format = null;
        int[] layerSizes = MlpBuilderLayerSizes.DEFAULT_GENERAL;
        int epochs = 400;
        float learningRate = 0.05f;
        int batchSize = 32;
        boolean help = false;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if ("--help".equals(arg) || "-h".equals(arg)) {
                help = true;
            } else if ("--profile".equals(arg) && i + 1 < args.length) {
                profile = WeightProfile.fromString(args[++i]);
            } else if ("--out".equals(arg) && i + 1 < args.length) {
                output = Path.of(args[++i]);
            } else if ("--format".equals(arg) && i + 1 < args.length) {
                format = parseFormat(args[++i]);
            } else if ("--layers".equals(arg) && i + 1 < args.length) {
                layerSizes = MlpBuilderLayerSizes.parse(args[++i]);
            } else if ("--epochs".equals(arg) && i + 1 < args.length) {
                epochs = Integer.parseInt(args[++i]);
            } else if ("--lr".equals(arg) && i + 1 < args.length) {
                learningRate = Float.parseFloat(args[++i]);
            } else if ("--batch".equals(arg) && i + 1 < args.length) {
                batchSize = Integer.parseInt(args[++i]);
            } else if (!arg.startsWith("-")) {
                output = Path.of(arg);
            } else {
                throw new IllegalArgumentException("unknown flag: " + arg);
            }
        }

        if (output == null) {
            output = profile == WeightProfile.ALGEBRA
                    ? Path.of("java/resources/models/algebra-v1.safetensors")
                    : Path.of("model.safetensors");
        }
        if (format == null) {
            format = WeightFormat.fromPath(output);
        }
        if (profile == WeightProfile.ALGEBRA) {
            layerSizes = MlpBuilderLayerSizes.ALGEBRA;
        }

        return new GenerateWeightsOptions(profile, output, format, layerSizes, epochs, learningRate, batchSize, help);
    }

    private static WeightFormat parseFormat(String value) {
        if ("safetensors".equalsIgnoreCase(value) || "safe".equalsIgnoreCase(value)) {
            return WeightFormat.SAFETENSORS;
        }
        if ("t4j_json".equalsIgnoreCase(value) || "json".equalsIgnoreCase(value)) {
            return WeightFormat.T4J_JSON;
        }
        throw new IllegalArgumentException("unknown format: " + value);
    }

    /** Layer size constants and parsing (keeps {@link com.github.tensor4j.nn.MlpBuilder} out of option wiring). */
    static final class MlpBuilderLayerSizes {
        static final int[] DEFAULT_GENERAL = new int[] {4, 8, 4, 1};
        static final int[] ALGEBRA = new int[] {3, 16, 16, 1};

        private MlpBuilderLayerSizes() {
        }

        static int[] parse(String csv) {
            return com.github.tensor4j.nn.MlpBuilder.parseLayerSizes(csv);
        }
    }

    static void printHelp() {
        System.out.println("GenerateWeights — tinygrad-compatible state_dict export");
        System.out.println();
        System.out.println("  general (default): build MLP from --layers, export init weights via get_state_dict");
        System.out.println("  algebra:           train AlgebraModel head, export bundled regression weights");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  GenerateWeights [--profile general|algebra] [--out PATH] [--format safetensors|t4j_json]");
        System.out.println("                  [--layers 4,8,4,1]   # general profile only");
        System.out.println("                  [--epochs N] [--lr F] [--batch N]  # algebra profile training");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  GenerateWeights --out model.safetensors");
        System.out.println("  GenerateWeights --layers 3,16,16,1 --out scratch.safetensors");
        System.out.println("  GenerateWeights --profile algebra --out java/resources/models/algebra-v1.safetensors");
    }
}
