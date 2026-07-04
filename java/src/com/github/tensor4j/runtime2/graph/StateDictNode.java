/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime2.graph;

import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.runtime2.state.StateDictWeight;

/** Single node in a tinygrad-style forward graph with deferred weight realize. */
public final class StateDictNode {

    private final String name;
    private final StateDictOp op;
    private final String inputA;
    private final String inputB;
    private final InferTensor weight;
    private final StateDictWeight lazyWeight;
    private final float eps;

    private StateDictNode(
            String name,
            StateDictOp op,
            String inputA,
            String inputB,
            InferTensor weight,
            StateDictWeight lazyWeight,
            float eps) {
        this.name = name;
        this.op = op;
        this.inputA = inputA;
        this.inputB = inputB;
        this.weight = weight;
        this.lazyWeight = lazyWeight;
        this.eps = eps;
    }

    public static StateDictNode input(String name) {
        return new StateDictNode(name, StateDictOp.INPUT, null, null, null, null, 0.0f);
    }

    public static StateDictNode rmsNorm(String name, String input, InferTensor weight, float eps) {
        return new StateDictNode(name, StateDictOp.RMS_NORM, input, null, weight, null, eps);
    }

    public static StateDictNode rmsNorm(String name, String input, StateDictWeight weight, float eps) {
        return new StateDictNode(name, StateDictOp.RMS_NORM, input, null, null, weight, eps);
    }

    public static StateDictNode mulMat(String name, String input, InferTensor weight) {
        return new StateDictNode(name, StateDictOp.MUL_MAT, input, null, weight, null, 0.0f);
    }

    public static StateDictNode mulMat(String name, String input, StateDictWeight weight) {
        return new StateDictNode(name, StateDictOp.MUL_MAT, input, null, null, weight, 0.0f);
    }

    public static StateDictNode add(String name, String a, String b) {
        return new StateDictNode(name, StateDictOp.ADD, a, b, null, null, 0.0f);
    }

    public static StateDictNode silu(String name, String input) {
        return new StateDictNode(name, StateDictOp.SILU, input, null, null, null, 0.0f);
    }

    public static StateDictNode swiglu(String name, String gate, String up) {
        return new StateDictNode(name, StateDictOp.SWIGLU, gate, up, null, null, 0.0f);
    }

    public static StateDictNode mul(String name, String a, InferTensor weight) {
        return new StateDictNode(name, StateDictOp.MUL, a, null, weight, null, 0.0f);
    }

    public String name() {
        return name;
    }

    public StateDictOp op() {
        return op;
    }

    public String inputA() {
        return inputA;
    }

    public String inputB() {
        return inputB;
    }

    public InferTensor weight() {
        return weight;
    }

    public StateDictWeight lazyWeight() {
        return lazyWeight;
    }

    public float eps() {
        return eps;
    }

    /** Materialize weight at graph eval (tinygrad {@code realize()} on dequant UOp). */
    InferTensor resolvedWeight() {
        if (lazyWeight != null) {
            return lazyWeight.tensor();
        }
        return weight;
    }
}
