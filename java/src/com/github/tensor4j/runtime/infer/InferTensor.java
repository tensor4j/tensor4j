/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.infer;

import com.github.tensor4j.runtime.ggml.GgmlTensorShape;

/** Row-major F32 inference buffer (ggml F32, tokens x features layout). */
public final class InferTensor {

    private final float[] data;
    private final int rows;
    private final int cols;

    private InferTensor(float[] data, int rows, int cols) {
        if (data.length != rows * cols) {
            throw new IllegalArgumentException("data length " + data.length + " != " + rows + "x" + cols);
        }
        this.data = data;
        this.rows = rows;
        this.cols = cols;
    }

    public static InferTensor zeros(int rows, int cols) {
        return new InferTensor(new float[rows * cols], rows, cols);
    }

    public static InferTensor of(float[] data, int rows, int cols) {
        return new InferTensor(data.clone(), rows, cols);
    }

    public static InferTensor vector(float... values) {
        return new InferTensor(values.clone(), 1, values.length);
    }

    public static InferTensor matrix(int rows, int cols, float... rowMajor) {
        return new InferTensor(rowMajor.clone(), rows, cols);
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public float[] data() {
        return data;
    }

    public float get(int row, int col) {
        return data[row * cols + col];
    }

    public void set(int row, int col, float value) {
        data[row * cols + col] = value;
    }

    public InferTensor copy() {
        return new InferTensor(data.clone(), rows, cols);
    }

    public GgmlTensorShape ggmlShape2d() {
        return GgmlTensorShape.of(cols, rows);
    }
}
