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

/**
 * Maps tinygrad op names to tensor4j coverage (layer 1 forward / layer 2 autograd).
 * Expected values use standard Java {@code float} math for cross-platform consistency.
 */
public final class OpCatalog {

    /** tinygrad-style ops with forward tests in {@code AllOpsForwardTest}. */
    public static final String[] LAYER1_OPS = {
            "zeros",
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "matmul",
            "relu",
            "sum",
            "mean",
            "sum_axis",
            "reshape",
            "flatten",
            "expand",
            "permute",
            "transpose",
            "contiguous",
            "detach",
            "pow2",
    };

    /** Ops with autograd tests in {@code AllOpsAutogradTest} / {@code GradCheckTest}. */
    public static final String[] LAYER2_OPS = {
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "matmul",
            "relu",
            "sum",
            "mean",
            "pow2",
            "reshape",
            "expand",
            "permute",
            "bias_add",
    };

    private OpCatalog() {
    }
}
