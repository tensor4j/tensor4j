/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.cli;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.io.ModelLoader;
import com.github.tensor4j.io.WeightFormat;
import com.github.tensor4j.models.algebra.AlgebraModel;
import com.github.tensor4j.models.algebra.AlgebraTrainer;
import com.github.tensor4j.models.algebra.AlgebraModel.AlgebraResult;

public final class CliRunner {

    private final PrintStream out;
    private final PrintStream err;
    private final AlgebraModel model;

    public CliRunner(AlgebraModel model, PrintStream out, PrintStream err) {
        this.model = model;
        this.out = out;
        this.err = err;
    }

    public int run(String[] args) {
        if (args.length == 0 || isHelp(args)) {
            printHelp();
            return 0;
        }
        String command = args[0].toLowerCase(Locale.ROOT);
        try {
            if ("info".equals(command)) {
                return info();
            }
            if ("infer".equals(command)) {
                return infer(parseOption(args, "--equation", "2x + 3 = 11"));
            }
            if ("train".equals(command)) {
                return train(
                        Integer.parseInt(parseOption(args, "--epochs", "40")),
                        Float.parseFloat(parseOption(args, "--lr", "0.05")),
                        Integer.parseInt(parseOption(args, "--batch", "16")));
            }
            if ("tensor".equals(command)) {
                return tensor(args);
            }
            if ("export".equals(command)) {
                String outPath = parseOption(args, "--out", "algebra-v1.safetensors");
                return exportModel(outPath, parseWeightFormat(args, Path.of(outPath)));
            }
            if ("gui".equals(command)) {
                out.println("GUI mode: launch without CLI arguments or use the desktop window.");
                return 0;
            }
            err.println("Unknown command: " + command);
            printHelp();
            return 2;
        } catch (Exception ex) {
            err.println("ERROR: " + ex.getMessage());
            return 1;
        }
    }

    private int info() {
        out.println("tensor4j — float32 tensor runtime for Tensor4j");
        out.println("Concepts: tensor · autograd · gradient flow · manifold · inference");
        out.println("Models:   algebra (high-school linear equations ax + b = c)");
        out.println("Compat:   tinygrad-compatible .t4j.json and .safetensors weight bundles");
        return 0;
    }

    private int infer(String equation) throws IOException {
        model.loadBundledWeights();
        AlgebraResult result = model.infer(equation);
        out.printf(Locale.US, "equation:  %s%n", equation);
        out.printf(Locale.US, "predicted: %.4f%n", result.predicted());
        out.printf(Locale.US, "exact:     %.4f%n", result.exact());
        out.printf(Locale.US, "abs error: %.6f%n", result.error());
        return 0;
    }

    private int train(int epochs, float lr, int batch) {
        AlgebraTrainer trainer = new AlgebraTrainer(model);
        List<Float> losses = trainer.train(epochs, lr, batch);
        out.printf(Locale.US, "trained algebra model: epochs=%d lr=%.4f batch=%d%n", epochs, lr, batch);
        out.printf(Locale.US, "initial loss: %.6f  final loss: %.6f%n",
                losses.get(0), losses.get(losses.size() - 1));
        return 0;
    }

    private int tensor(String[] args) {
        String shapeRaw = parseOption(args, "--shape", "2,2");
        int[] shape = parseIntList(shapeRaw);
        String dataRaw = parseOption(args, "--data", "");
        Tensor tensor;
        if (dataRaw.isBlank()) {
            tensor = Tensor.randn(shape);
        } else {
            String[] parts = dataRaw.split(",");
            float[] values = new float[parts.length];
            for (int i = 0; i < parts.length; i++) {
                values[i] = Float.parseFloat(parts[i].trim());
            }
            tensor = Tensor.of(values, shape);
        }
        out.println("tensor " + tensor);
        return 0;
    }

    private int exportModel(String outPath, WeightFormat format) throws IOException {
        Path path = Path.of(outPath);
        ModelLoader.save(path, ModelLoader.exportTensors(model.network()), format);
        out.println("exported " + format + " weights: " + path.toAbsolutePath());
        return 0;
    }

    private static WeightFormat parseWeightFormat(String[] args, Path outPath) {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--format".equals(args[i])) {
                return parseWeightFormatValue(args[i + 1]);
            }
        }
        return WeightFormat.fromPath(outPath);
    }

    private static WeightFormat parseWeightFormatValue(String value) {
        if ("safetensors".equalsIgnoreCase(value) || "safe".equalsIgnoreCase(value)) {
            return WeightFormat.SAFETENSORS;
        }
        if ("t4j_json".equalsIgnoreCase(value) || "json".equalsIgnoreCase(value)) {
            return WeightFormat.T4J_JSON;
        }
        return WeightFormat.SAFETENSORS;
    }

    private static int[] parseIntList(String csv) {
        String[] parts = csv.split(",");
        int[] values = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Integer.parseInt(parts[i].trim());
        }
        return values;
    }

    private static String parseOption(String[] args, String flag, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (flag.equals(args[i])) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }

    private static boolean isHelp(String[] args) {
        return Arrays.asList(args).contains("--help") || Arrays.asList(args).contains("-h");
    }

    private void printHelp() {
        out.println("tensor4j — usage");
        out.println("  tensor4j info");
        out.println("  tensor4j infer --equation \"2x + 3 = 11\"");
        out.println("  tensor4j train [--epochs 40] [--lr 0.05] [--batch 16]");
        out.println("  tensor4j tensor [--shape 2,2] [--data 1,2,3,4]");
        out.println("  tensor4j export --out algebra-v1.safetensors [--format safetensors|t4j_json]");
        out.println("  tensor4j gui");
        out.println("  tensor4j --help");
    }
}
