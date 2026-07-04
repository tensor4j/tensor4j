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
import com.github.tensor4j.runtime.infer.InferWeight;
import com.github.tensor4j.runtime.infer.RopeConfig;
import com.github.tensor4j.runtime.memory.KvCacheStore;

/** One decoder layer: pre-norm attention + residual + pre-norm SwiGLU FFN + residual. */
public final class LlamaBlockForward {

    public record Weights(
            InferWeight attnNorm,
            InferWeight wq,
            InferWeight wk,
            InferWeight wv,
            InferWeight wo,
            InferWeight ffnNorm,
            InferWeight wGate,
            InferWeight wUp,
            InferWeight wDown) {
    }

    private LlamaBlockForward() {
    }

    public static InferTensor forward(
            InferTensor x,
            Weights weights,
            KvCacheStore cache,
            int nHead,
            int nHeadKv,
            float rmsEps,
            int[] positions,
            RopeConfig rope) {
        InferTensor attnOut = LlamaAttentionForward.forward(
                x,
                weights.attnNorm().tensor(),
                weights.wq().tensor(),
                weights.wk().tensor(),
                weights.wv().tensor(),
                weights.wo().tensor(),
                cache,
                nHead,
                nHeadKv,
                rmsEps,
                positions,
                rope,
                true);
        InferTensor afterAttn = GgmlOps.add(x, attnOut);
        InferTensor ffnIn = GgmlOps.rmsNorm(afterAttn, weights.ffnNorm().tensor(), rmsEps);
        InferTensor gate = GgmlOps.mulMat(ffnIn, weights.wGate().tensor());
        InferTensor up = GgmlOps.mulMat(ffnIn, weights.wUp().tensor());
        InferTensor ffnMid = GgmlOps.swiglu(gate, up);
        InferTensor ffnOut = GgmlOps.mulMat(ffnMid, weights.wDown().tensor());
        return GgmlOps.add(afterAttn, ffnOut);
    }
}
