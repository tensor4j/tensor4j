/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.algebra;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Parsed linear equation {@code ax + b = c} for high-school algebra demos. */
public record AlgebraEquation(float a, float b, float c, String normalized) {

    private static final Pattern EQUATION = Pattern.compile(
            "^\\s*(-?\\d*\\.?\\d*)\\s*\\*?\\s*[xX]\\s*([+-]\\s*\\d*\\.?\\d*)?\\s*=\\s*(-?\\d*\\.?\\d*)\\s*$");

    public static AlgebraEquation parse(String input) {
        String compact = input.replace(" ", "").toLowerCase(Locale.ROOT);
        Matcher matcher = EQUATION.matcher(compact);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("unsupported equation format: " + input
                    + " (expected ax + b = c, e.g. 2x + 3 = 11)");
        }
        float a = parseCoefficient(matcher.group(1), 1f);
        float b = parseOffset(matcher.group(2));
        float c = Float.parseFloat(matcher.group(3));
        if (a == 0f) {
            throw new IllegalArgumentException("coefficient of x must be non-zero");
        }
        return new AlgebraEquation(a, b, c, compact);
    }

    public float exactSolution() {
        return (c - b) / a;
    }

    public float[] features() {
        return new float[] {a, b, c};
    }

    /** Scaled coefficients for stable float32 training/inference. */
    public float[] normalizedFeatures() {
        return new float[] {a / 10f, b / 10f, c / 10f};
    }

    private static float parseCoefficient(String token, float defaultValue) {
        if (token == null || token.isEmpty() || token.equals("+")) {
            return defaultValue;
        }
        if (token.equals("-")) {
            return -defaultValue;
        }
        return Float.parseFloat(token);
    }

    private static float parseOffset(String token) {
        if (token == null || token.isBlank()) {
            return 0f;
        }
        return Float.parseFloat(token.replace(" ", ""));
    }
}
