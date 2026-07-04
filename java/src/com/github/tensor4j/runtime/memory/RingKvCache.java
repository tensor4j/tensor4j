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
 * Ring-buffer KV cache with oldest-token eviction (llama-kv-cache bounded storage).
 * Logical token 0 is always the oldest retained slot.
 */
public final class RingKvCache implements KvCacheStore {

    private final int maxSeq;
    private final int nHeadKv;
    private final int headDim;
    private final int rowWidth;
    private final float[] kData;
    private final float[] vData;
    private int start;
    private int count;

    public RingKvCache(int maxSeq, int nHeadKv, int headDim) {
        if (maxSeq <= 0 || nHeadKv <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("invalid kv cache shape");
        }
        this.maxSeq = maxSeq;
        this.nHeadKv = nHeadKv;
        this.headDim = headDim;
        this.rowWidth = nHeadKv * headDim;
        this.kData = new float[maxSeq * rowWidth];
        this.vData = new float[maxSeq * rowWidth];
        this.start = 0;
        this.count = 0;
    }

    @Override
    public int nKv() {
        return count;
    }

    @Override
    public int maxSeq() {
        return maxSeq;
    }

    @Override
    public void clear() {
        start = 0;
        count = 0;
    }

    @Override
    public void append(InferTensor kCur, InferTensor vCur) {
        if (kCur.rows() != 1 || vCur.rows() != 1) {
            throw new IllegalArgumentException("append expects single-token rows");
        }
        validateWidth(kCur, vCur);
        int physical = nextPhysicalSlot();
        int offset = physical * rowWidth;
        System.arraycopy(kCur.data(), 0, kData, offset, rowWidth);
        System.arraycopy(vCur.data(), 0, vData, offset, rowWidth);
    }

    @Override
    public void appendBlock(InferTensor kBlock, InferTensor vBlock) {
        if (kBlock.rows() != vBlock.rows()) {
            throw new IllegalArgumentException("k/v row mismatch");
        }
        validateWidth(kBlock, vBlock);
        for (int r = 0; r < kBlock.rows(); r++) {
            int offset = r * rowWidth;
            float[] kRow = new float[rowWidth];
            float[] vRow = new float[rowWidth];
            System.arraycopy(kBlock.data(), offset, kRow, 0, rowWidth);
            System.arraycopy(vBlock.data(), offset, vRow, 0, rowWidth);
            append(InferTensor.of(kRow, 1, rowWidth), InferTensor.of(vRow, 1, rowWidth));
        }
    }

    @Override
    public InferTensor keysForHead(int head) {
        return gatherHead(kData, head);
    }

    @Override
    public InferTensor valuesForHead(int head) {
        return gatherHead(vData, head);
    }

    private int nextPhysicalSlot() {
        if (count < maxSeq) {
            int physical = (start + count) % maxSeq;
            count++;
            return physical;
        }
        start = (start + 1) % maxSeq;
        return (start + maxSeq - 1) % maxSeq;
    }

    private InferTensor gatherHead(float[] data, int head) {
        if (head < 0 || head >= nHeadKv) {
            throw new IllegalArgumentException("head " + head);
        }
        float[] out = new float[count * headDim];
        for (int logical = 0; logical < count; logical++) {
            int physical = (start + logical) % maxSeq;
            int src = physical * rowWidth + head * headDim;
            int dst = logical * headDim;
            System.arraycopy(data, src, out, dst, headDim);
        }
        return InferTensor.of(out, count, headDim);
    }

    private void validateWidth(InferTensor k, InferTensor v) {
        if (k.cols() != rowWidth || v.cols() != rowWidth) {
            throw new IllegalArgumentException("expected width " + rowWidth);
        }
    }
}
