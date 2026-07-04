/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core;

/** Contiguous float32 kernels (tinygrad realize path at sample scale). */
public final class TensorMath {

    private TensorMath() {
    }

    static float[] matmul2d(float[] left, int rows, int inner, float[] right, int cols) {
        float[] result = new float[rows * cols];
        matmul2dInto(result, left, rows, inner, right, cols);
        return result;
    }

    /** {@code C[m,n] = A[m,k] @ B[k,n]} with i-k-j loop order for cache locality. */
    static void matmul2dInto(float[] result, float[] left, int rows, int inner, float[] right, int cols) {
        for (int r = 0; r < rows; r++) {
            int rowBase = r * cols;
            int leftRow = r * inner;
            for (int k = 0; k < inner; k++) {
                float a = left[leftRow + k];
                int rightRow = k * cols;
                for (int c = 0; c < cols; c++) {
                    result[rowBase + c] += a * right[rightRow + c];
                }
            }
        }
    }

    static float[] transpose2d(float[] matrix, int rows, int cols) {
        float[] result = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[c * rows + r] = matrix[r * cols + c];
            }
        }
        return result;
    }

    /** {@code dA = dY @ B^T} for row-major 2D matrices. */
    public static float[] matmulGradLeft(float[] gradOut, int rows, int inner, int cols, float[] right, int rightRows, int rightCols) {
        float[] rightTransposed = transpose2d(right, rightRows, rightCols);
        return matmul2d(gradOut, rows, cols, rightTransposed, inner);
    }

    /** {@code dB = A^T @ dY}. */
    public static float[] matmulGradRight(float[] left, int leftRows, int leftCols, float[] gradOut, int rows, int cols) {
        float[] leftTransposed = transpose2d(left, leftRows, leftCols);
        return matmul2d(leftTransposed, leftCols, leftRows, gradOut, cols);
    }

    static void addInto(float[] out, float[] left, float[] right, int n) {
        for (int i = 0; i < n; i++) {
            out[i] = left[i] + right[i];
        }
    }

    static void mulInto(float[] out, float[] left, float[] right, int n) {
        for (int i = 0; i < n; i++) {
            out[i] = left[i] * right[i];
        }
    }

    static void divInto(float[] out, float[] left, float[] right, int n) {
        for (int i = 0; i < n; i++) {
            out[i] = left[i] / right[i];
        }
    }

    static void reluInto(float[] out, float[] input, int n) {
        for (int i = 0; i < n; i++) {
            out[i] = Math.max(0f, input[i]);
        }
    }

    static void negInto(float[] out, float[] input, int n) {
        for (int i = 0; i < n; i++) {
            out[i] = -input[i];
        }
    }

    static void axpy(float[] params, float[] grad, int n, float learningRate) {
        for (int i = 0; i < n; i++) {
            params[i] -= learningRate * grad[i];
        }
    }

    public static float l2Norm(float[] values, int n) {
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            sum += values[i] * values[i];
        }
        return (float) Math.sqrt(sum);
    }

    public static void scale(float[] values, int n, float factor) {
        for (int i = 0; i < n; i++) {
            values[i] *= factor;
        }
    }

    static float mean(float[] values, int n) {
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            sum += values[i];
        }
        return sum / n;
    }
}
