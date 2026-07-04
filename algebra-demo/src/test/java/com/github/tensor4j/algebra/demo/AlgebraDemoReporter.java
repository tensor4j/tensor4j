/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.algebra.demo;

import java.util.List;
import java.util.Locale;
import com.github.tensor4j.models.algebra.AlgebraModel.AlgebraResult;

/** Prints a readable demo transcript to the Failsafe test log. */
final class AlgebraDemoReporter {

    private AlgebraDemoReporter() {
    }

    static void banner(String title) {
        System.out.println();
        System.out.println("=== tensor4j algebra-demo: " + title + " ===");
    }

    static void result(AlgebraResult result) {
        System.out.printf(Locale.US, "  equation:  %s%n", result.equation().normalized());
        System.out.println("  worked:");
        List<String> steps = WorkedSolution.steps(result.equation());
        for (int i = 0; i < steps.size(); i++) {
            System.out.println("             " + steps.get(i));
        }
        System.out.printf(
                Locale.US,
                "  mlp:       predicted x = %8.4f   (exact x = %8.4f, abs error = %.4f)%n",
                result.predicted(),
                result.exact(),
                result.error());
    }

    static void summary(int passed, int total, float maxError) {
        System.out.printf(
                Locale.US,
                "=== algebra-demo pass: %d/%d (max abs error <= %.2f) ===%n%n",
                passed,
                total,
                maxError);
    }
}
