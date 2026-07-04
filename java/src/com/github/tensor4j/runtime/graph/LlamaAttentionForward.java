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

import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.runtime.infer.Rope;
import com.github.tensor4j.runtime.infer.RopeConfig;
import com.github.tensor4j.runtime.memory.KvCacheStore;

/**
 * Multi-head attention forward with KV cache (llama-graph build_qkv + cache cpy_k/cpy_v + RoPE).
 * GQA: {@code wk}/{@code wv} are [n_embd, n_head_kv * head_dim]; {@code wq} is [n_embd, n_embd].
 */
public final class LlamaAttentionForward {

    private LlamaAttentionForward() {
    }

    public static InferTensor forward(
            InferTensor x,
            InferTensor attnNormWeight,
            InferTensor wq,
            InferTensor wk,
            InferTensor wv,
            InferTensor wo,
            KvCacheStore cache,
            int nHead,
            int nHeadKv,
            float rmsEps,
            int[] positions,
            RopeConfig rope,
            boolean causal) {
        validateHeads(nHead, nHeadKv);
        int nEmbd = x.cols();
        int headDim = nEmbd / nHead;
        int kvWidth = nHeadKv * headDim;
        validateWeights(nEmbd, headDim, kvWidth, wq, wk, wv, wo);

        InferTensor normed = GgmlOps.rmsNorm(x, attnNormWeight, rmsEps);
        InferTensor q = GgmlOps.mulMatOut(normed, wq);
        InferTensor kCur = GgmlOps.mulMatOut(normed, wk);
        InferTensor vCur = GgmlOps.mulMatOut(normed, wv);

        if (positions == null) {
            positions = defaultPositions(x.rows(), cache.nKv());
        }
        q = Rope.applyHeads(q, nHead, headDim, positions, rope);
        kCur = Rope.applyHeads(kCur, nHeadKv, headDim, positions, rope);

        int pastKv = cache.nKv();
        if (kCur.rows() == 1) {
            cache.append(kCur, vCur);
        } else {
            cache.appendBlock(kCur, vCur);
        }

        float scale = (float) (1.0 / Math.sqrt(headDim));
        InferTensor[] headOut = new InferTensor[nHead];
        int qHeadsPerKv = nHead / nHeadKv;
        for (int h = 0; h < nHead; h++) {
            int kvHead = h / qHeadsPerKv;
            InferTensor qHead = sliceHead(q, h, headDim, nEmbd);
            InferTensor kHead = cache.keysForHead(kvHead);
            InferTensor vHead = cache.valuesForHead(kvHead);
            InferTensor scores = GgmlOps.scale(GgmlOps.qkScores(qHead, kHead), scale);
            if (causal) {
                scores = GgmlOps.applyCausalMask(scores, pastKv);
            }
            InferTensor probs = GgmlOps.softmaxRows(scores);
            headOut[h] = GgmlOps.attnContext(probs, vHead);
        }
        InferTensor merged = mergeHeads(headOut, headDim);
        return GgmlOps.mulMatOut(merged, wo);
    }

    private static int[] defaultPositions(int rows, int pastKv) {
        int[] positions = new int[rows];
        for (int i = 0; i < rows; i++) {
            positions[i] = pastKv + i;
        }
        return positions;
    }

    private static void validateHeads(int nHead, int nHeadKv) {
        if (nHead <= 0 || nHeadKv <= 0 || nHead % nHeadKv != 0) {
            throw new IllegalArgumentException("n_head " + nHead + " must be multiple of n_head_kv " + nHeadKv);
        }
    }

    private static void validateWeights(int nEmbd, int headDim, int kvWidth, InferTensor wq, InferTensor wk,
            InferTensor wv, InferTensor wo) {
        requireMatrix(wq, nEmbd, nEmbd, "wq");
        requireMatrix(wk, kvWidth, nEmbd, "wk");
        requireMatrix(wv, kvWidth, nEmbd, "wv");
        requireMatrix(wo, nEmbd, nEmbd, "wo");
        if (headDim * (nEmbd / headDim) != nEmbd) {
            throw new IllegalArgumentException("invalid embedding width " + nEmbd);
        }
    }

    private static void requireMatrix(InferTensor weight, int rows, int cols, String name) {
        if (weight.rows() != rows || weight.cols() != cols) {
            throw new IllegalArgumentException(name + " expected " + rows + "x" + cols + " but was "
                    + weight.rows() + "x" + weight.cols());
        }
    }

    private static InferTensor sliceHead(InferTensor q, int head, int headDim, int nEmbd) {
        float[] out = new float[q.rows() * headDim];
        float[] qd = q.data();
        for (int t = 0; t < q.rows(); t++) {
            int src = t * nEmbd + head * headDim;
            int dst = t * headDim;
            System.arraycopy(qd, src, out, dst, headDim);
        }
        return InferTensor.of(out, q.rows(), headDim);
    }

    private static InferTensor mergeHeads(InferTensor[] heads, int headDim) {
        int nTokens = heads[0].rows();
        int nHead = heads.length;
        float[] out = new float[nTokens * nHead * headDim];
        for (int h = 0; h < nHead; h++) {
            float[] hd = heads[h].data();
            for (int t = 0; t < nTokens; t++) {
                int dst = t * nHead * headDim + h * headDim;
                int src = t * headDim;
                System.arraycopy(hd, src, out, dst, headDim);
            }
        }
        return InferTensor.of(out, nTokens, nHead * headDim);
    }
}
