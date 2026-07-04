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

/** F32 inference kernels (ggml-cpu ops.cpp — mul_mat, rms_norm, silu, soft_max). */
public final class GgmlOps {

    private GgmlOps() {
    }

    /** {@code ggml_mul_mat}: [m,k] @ [k,n] -> [m,n] row-major. */
    public static InferTensor mulMat(InferTensor a, InferTensor b) {
        int m = a.rows();
        int k = a.cols();
        int n = b.cols();
        if (b.rows() != k) {
            throw new IllegalArgumentException("mulMat shape mismatch: " + a.rows() + "x" + a.cols()
                    + " @ " + b.rows() + "x" + b.cols());
        }
        float[] out = new float[m * n];
        float[] ad = a.data();
        float[] bd = b.data();
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int outRow = i * n;
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += ad[aRow + p] * bd[p * n + j];
                }
                out[outRow + j] = sum;
            }
        }
        return InferTensor.of(out, m, n);
    }

    public static InferTensor add(InferTensor a, InferTensor b) {
        requireSameShape(a, b);
        float[] out = new float[a.data().length];
        float[] ad = a.data();
        float[] bd = b.data();
        for (int i = 0; i < out.length; i++) {
            out[i] = ad[i] + bd[i];
        }
        return InferTensor.of(out, a.rows(), a.cols());
    }

    /** Element-wise multiply; {@code b} may be a 1 x cols weight row broadcast across rows. */
    public static InferTensor mul(InferTensor a, InferTensor b) {
        if (a.rows() == b.rows() && a.cols() == b.cols()) {
            float[] out = new float[a.data().length];
            float[] ad = a.data();
            float[] bd = b.data();
            for (int i = 0; i < out.length; i++) {
                out[i] = ad[i] * bd[i];
            }
            return InferTensor.of(out, a.rows(), a.cols());
        }
        if (b.rows() == 1 && b.cols() == a.cols()) {
            float[] out = new float[a.data().length];
            float[] ad = a.data();
            float[] bd = b.data();
            int cols = a.cols();
            for (int r = 0; r < a.rows(); r++) {
                int offset = r * cols;
                for (int c = 0; c < cols; c++) {
                    out[offset + c] = ad[offset + c] * bd[c];
                }
            }
            return InferTensor.of(out, a.rows(), a.cols());
        }
        throw new IllegalArgumentException("mul broadcast unsupported for " + a.rows() + "x" + a.cols()
                + " and " + b.rows() + "x" + b.cols());
    }

    /** {@code ggml_rms_norm} per row, optional weight multiply (llama build_norm). */
    public static InferTensor rmsNorm(InferTensor x, InferTensor weight, float eps) {
        if (weight != null && (weight.rows() != 1 || weight.cols() != x.cols())) {
            throw new IllegalArgumentException("weight must be 1 x " + x.cols());
        }
        float[] out = new float[x.data().length];
        float[] xd = x.data();
        float[] wd = weight == null ? null : weight.data();
        int cols = x.cols();
        for (int r = 0; r < x.rows(); r++) {
            int offset = r * cols;
            double sumSq = 0.0;
            for (int c = 0; c < cols; c++) {
                float v = xd[offset + c];
                sumSq += (double) v * v;
            }
            float scale = (float) (1.0 / Math.sqrt(sumSq / cols + eps));
            for (int c = 0; c < cols; c++) {
                float v = xd[offset + c] * scale;
                if (wd != null) {
                    v *= wd[c];
                }
                out[offset + c] = v;
            }
        }
        return InferTensor.of(out, x.rows(), x.cols());
    }

    /** {@code ggml_silu}: x / (1 + exp(-x)). */
    public static InferTensor silu(InferTensor x) {
        float[] out = new float[x.data().length];
        float[] xd = x.data();
        for (int i = 0; i < xd.length; i++) {
            float v = xd[i];
            out[i] = v / (1.0f + (float) Math.exp(-v));
        }
        return InferTensor.of(out, x.rows(), x.cols());
    }

    /** SwiGLU gate: silu(gate) * up (llama build_ffn LLM_FFN_SWIGLU). */
    public static InferTensor swiglu(InferTensor gate, InferTensor up) {
        return mul(silu(gate), up);
    }

    /** {@code ggml_soft_max} on each row (causal mask applied externally). */
    public static InferTensor softmaxRows(InferTensor scores) {
        float[] out = new float[scores.data().length];
        float[] sd = scores.data();
        int cols = scores.cols();
        for (int r = 0; r < scores.rows(); r++) {
            int offset = r * cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) {
                max = Math.max(max, sd[offset + c]);
            }
            double sum = 0.0;
            for (int c = 0; c < cols; c++) {
                float e = (float) Math.exp(sd[offset + c] - max);
                out[offset + c] = e;
                sum += e;
            }
            float inv = (float) (1.0 / sum);
            for (int c = 0; c < cols; c++) {
                out[offset + c] *= inv;
            }
        }
        return InferTensor.of(out, scores.rows(), scores.cols());
    }

    /** Scale a row vector by 1/sqrt(headDim) before QK^T (attention). */
    public static InferTensor scale(InferTensor x, float factor) {
        float[] out = new float[x.data().length];
        float[] xd = x.data();
        for (int i = 0; i < xd.length; i++) {
            out[i] = xd[i] * factor;
        }
        return InferTensor.of(out, x.rows(), x.cols());
    }

    /** Q [n_t,d] x K [n_k,d]^T -> scores [n_t,n_k]. */
    public static InferTensor qkScores(InferTensor q, InferTensor k) {
        int nT = q.rows();
        int nK = k.rows();
        int d = q.cols();
        if (k.cols() != d) {
            throw new IllegalArgumentException("qk dim mismatch");
        }
        float[] out = new float[nT * nK];
        float[] qd = q.data();
        float[] kd = k.data();
        for (int t = 0; t < nT; t++) {
            int qRow = t * d;
            int outRow = t * nK;
            for (int i = 0; i < nK; i++) {
                float sum = 0.0f;
                int kRow = i * d;
                for (int j = 0; j < d; j++) {
                    sum += qd[qRow + j] * kd[kRow + j];
                }
                out[outRow + i] = sum;
            }
        }
        return InferTensor.of(out, nT, nK);
    }

    /** weights [n_t,n_k] x V [n_k,d] -> context [n_t,d]. */
    public static InferTensor attnContext(InferTensor weights, InferTensor v) {
        return mulMat(weights, v);
    }

    private static void requireSameShape(InferTensor a, InferTensor b) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw new IllegalArgumentException("shape mismatch " + a.rows() + "x" + a.cols()
                    + " vs " + b.rows() + "x" + b.cols());
        }
    }
}
