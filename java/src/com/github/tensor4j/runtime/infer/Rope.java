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

/** Rotary position embedding (ggml_compute_forward_rope, GGML_ROPE_TYPE_NORMAL + YaRN). */
public final class Rope {

    private Rope() {
    }

    /**
     * Apply RoPE to {@code tensor} shaped [n_tokens, n_heads * head_dim].
     * Only the first {@code config.ropeDim()} features per head are rotated (partial RoPE).
     */
    public static InferTensor applyHeads(InferTensor tensor, int nHeads, int headDim, int[] positions,
            RopeConfig config) {
        if (!config.enabled()) {
            return tensor;
        }
        if (positions.length != tensor.rows()) {
            throw new IllegalArgumentException("positions length " + positions.length + " != rows " + tensor.rows());
        }
        int ropeDim = config.ropeDim();
        if (ropeDim > headDim || ropeDim % 2 != 0) {
            throw new IllegalArgumentException("invalid ropeDim " + ropeDim + " for headDim " + headDim);
        }
        float[] out = tensor.data().clone();
        int rowWidth = nHeads * headDim;
        float[] cosSin = new float[ropeDim];
        for (int t = 0; t < tensor.rows(); t++) {
            fillCosSin(cosSin, positions[t], config);
            int rowOffset = t * rowWidth;
            for (int h = 0; h < nHeads; h++) {
                rotatePairs(out, rowOffset + h * headDim, ropeDim, cosSin);
            }
        }
        return InferTensor.of(out, tensor.rows(), rowWidth);
    }

    private static void fillCosSin(float[] cosSin, int position, RopeConfig config) {
        float theta = position;
        float thetaScale = (float) Math.pow(config.freqBase(), -2.0 / config.ropeDim());
        float[] corrDims = yarnCorrDims(config);
        for (int i0 = 0; i0 < config.ropeDim(); i0 += 2) {
            float angle = computeAngle(theta, config, corrDims, i0);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            if (config.yarnEnabled()) {
                float mscale = yarnMscale(config);
                cos *= mscale;
                sin *= mscale;
            }
            cosSin[i0] = cos;
            cosSin[i0 + 1] = sin;
            theta *= thetaScale;
        }
    }

    private static float computeAngle(float thetaExtrap, RopeConfig config, float[] corrDims, int i0) {
        float thetaInterp = config.freqScale() * thetaExtrap;
        if (!config.yarnEnabled()) {
            return thetaInterp;
        }
        float ramp = ropeYarnRamp(corrDims[0], corrDims[1], i0) * config.yarnExtFactor();
        return thetaInterp * (1.0f - ramp) + thetaExtrap * ramp;
    }

    private static float yarnMscale(RopeConfig config) {
        float mscale = config.yarnAttnFactor();
        if (config.freqScale() != 0.0f) {
            mscale *= 1.0f + 0.1f * (float) Math.log(1.0 / config.freqScale());
        }
        return mscale;
    }

    /** {@code ggml_rope_yarn_corr_dims}. */
    static float[] yarnCorrDims(RopeConfig config) {
        int nDims = config.ropeDim();
        int nCtxOrig = config.yarnOrigCtx() > 0 ? config.yarnOrigCtx() : 512;
        float start = (float) Math.floor(yarnCorrDim(nDims, nCtxOrig, config.yarnBetaFast(), config.freqBase()));
        float end = (float) Math.ceil(yarnCorrDim(nDims, nCtxOrig, config.yarnBetaSlow(), config.freqBase()));
        float low = Math.max(0.0f, start);
        float high = Math.min(nDims - 1, end);
        return new float[] {low, high};
    }

    private static float yarnCorrDim(int nDims, int nCtxOrig, float beta, float base) {
        return (float) (nDims * Math.log(nCtxOrig / (beta * 2.0 * Math.PI)) / (2.0 * Math.log(base)));
    }

    private static float ropeYarnRamp(float low, float high, int i0) {
        float y = (i0 / 2.0f - low) / Math.max(0.001f, high - low);
        return 1.0f - Math.min(1.0f, Math.max(0.0f, y));
    }

    private static void rotatePairs(float[] data, int offset, int ropeDim, float[] cosSin) {
        for (int i0 = 0; i0 < ropeDim; i0 += 2) {
            float cos = cosSin[i0];
            float sin = cosSin[i0 + 1];
            float x0 = data[offset + i0];
            float x1 = data[offset + i0 + 1];
            data[offset + i0] = x0 * cos - x1 * sin;
            data[offset + i0 + 1] = x0 * sin + x1 * cos;
        }
    }
}
