/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.graph;

import com.github.tensor4j.runtime.infer.InferTensor;

/** Single node in a forward inference graph. */
public final class InferNode {

    private final String name;
    private final InferOp op;
    private final String inputA;
    private final String inputB;
    private final InferTensor weight;
    private final float eps;

    private InferNode(String name, InferOp op, String inputA, String inputB, InferTensor weight, float eps) {
        this.name = name;
        this.op = op;
        this.inputA = inputA;
        this.inputB = inputB;
        this.weight = weight;
        this.eps = eps;
    }

    public static InferNode input(String name) {
        return new InferNode(name, InferOp.INPUT, null, null, null, 0.0f);
    }

    public static InferNode rmsNorm(String name, String input, InferTensor weight, float eps) {
        return new InferNode(name, InferOp.RMS_NORM, input, null, weight, eps);
    }

    public static InferNode mulMat(String name, String input, InferTensor weight) {
        return new InferNode(name, InferOp.MUL_MAT, input, null, weight, 0.0f);
    }

    public static InferNode add(String name, String a, String b) {
        return new InferNode(name, InferOp.ADD, a, b, null, 0.0f);
    }

    public static InferNode silu(String name, String input) {
        return new InferNode(name, InferOp.SILU, input, null, null, 0.0f);
    }

    public static InferNode swiglu(String name, String gate, String up) {
        return new InferNode(name, InferOp.SWIGLU, gate, up, null, 0.0f);
    }

    public static InferNode mul(String name, String a, InferTensor weight) {
        return new InferNode(name, InferOp.MUL, a, null, weight, 0.0f);
    }

    public String name() {
        return name;
    }

    public InferOp op() {
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

    public float eps() {
        return eps;
    }
}
