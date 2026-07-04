/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.memory;

import com.github.tensor4j.runtime.infer.InferTensor;

/**
 * Simplified KV cache (llama-kv-cache.h).
 * Layout: K/V stored as [n_kv, n_head_kv * head_dim] row-major per layer slot.
 */
public final class KvCache implements KvCacheStore {

    private final int maxSeq;
    private final int nHeadKv;
    private final int headDim;
    private int nKv;
    private final float[] kData;
    private final float[] vData;

    public KvCache(int maxSeq, int nHeadKv, int headDim) {
        if (maxSeq <= 0 || nHeadKv <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("invalid kv cache shape");
        }
        this.maxSeq = maxSeq;
        this.nHeadKv = nHeadKv;
        this.headDim = headDim;
        this.nKv = 0;
        int rowWidth = nHeadKv * headDim;
        this.kData = new float[maxSeq * rowWidth];
        this.vData = new float[maxSeq * rowWidth];
    }

    @Override
    public int nKv() {
        return nKv;
    }

    @Override
    public int maxSeq() {
        return maxSeq;
    }

    public int nHeadKv() {
        return nHeadKv;
    }

    public int headDim() {
        return headDim;
    }

    @Override
    public void clear() {
        nKv = 0;
    }

    @Override
    public void append(InferTensor kCur, InferTensor vCur) {
        if (kCur.rows() != 1 || vCur.rows() != 1) {
            throw new IllegalArgumentException("append expects single-token rows");
        }
        int width = nHeadKv * headDim;
        if (kCur.cols() != width || vCur.cols() != width) {
            throw new IllegalArgumentException("expected width " + width);
        }
        if (nKv >= maxSeq) {
            throw new IllegalStateException("kv cache full");
        }
        System.arraycopy(kCur.data(), 0, kData, nKv * width, width);
        System.arraycopy(vCur.data(), 0, vData, nKv * width, width);
        nKv++;
    }

    @Override
    public void appendBlock(InferTensor kBlock, InferTensor vBlock) {
        if (kBlock.rows() != vBlock.rows()) {
            throw new IllegalArgumentException("k/v row mismatch");
        }
        int width = nHeadKv * headDim;
        if (kBlock.cols() != width || vBlock.cols() != width) {
            throw new IllegalArgumentException("expected width " + width);
        }
        if (nKv + kBlock.rows() > maxSeq) {
            throw new IllegalStateException("kv cache overflow");
        }
        System.arraycopy(kBlock.data(), 0, kData, nKv * width, kBlock.data().length);
        System.arraycopy(vBlock.data(), 0, vData, nKv * width, vBlock.data().length);
        nKv += kBlock.rows();
    }

    /** K matrix for attention: [n_kv, n_head_kv * head_dim]. */
    public InferTensor keys() {
        return view(kData, nKv, nHeadKv * headDim);
    }

    /** V matrix for attention: [n_kv, n_head_kv * head_dim]. */
    public InferTensor values() {
        return view(vData, nKv, nHeadKv * headDim);
    }

    @Override
    public InferTensor keysForHead(int head) {
        if (head < 0 || head >= nHeadKv) {
            throw new IllegalArgumentException("head " + head);
        }
        float[] out = new float[nKv * headDim];
        int width = nHeadKv * headDim;
        for (int t = 0; t < nKv; t++) {
            int src = t * width + head * headDim;
            int dst = t * headDim;
            System.arraycopy(kData, src, out, dst, headDim);
        }
        return InferTensor.of(out, nKv, headDim);
    }

    @Override
    public InferTensor valuesForHead(int head) {
        if (head < 0 || head >= nHeadKv) {
            throw new IllegalArgumentException("head " + head);
        }
        float[] out = new float[nKv * headDim];
        int width = nHeadKv * headDim;
        for (int t = 0; t < nKv; t++) {
            int src = t * width + head * headDim;
            int dst = t * headDim;
            System.arraycopy(vData, src, out, dst, headDim);
        }
        return InferTensor.of(out, nKv, headDim);
    }

    private static InferTensor view(float[] data, int rows, int cols) {
        float[] slice = new float[rows * cols];
        System.arraycopy(data, 0, slice, 0, slice.length);
        return InferTensor.of(slice, rows, cols);
    }
}
