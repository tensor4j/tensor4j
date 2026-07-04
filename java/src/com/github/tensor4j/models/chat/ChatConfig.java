/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat;

import com.github.tensor4j.runtime.gguf.GgufHeader;
import com.github.tensor4j.runtime.gguf.GgufKvEntry;
import com.github.tensor4j.runtime.infer.RopeConfig;
import com.github.tensor4j.runtime.infer.RopeScalingType;

/** Llama hyperparameters from GGUF KV metadata (llama.cpp hparams). */
public record ChatConfig(
        int nVocab,
        int nEmbd,
        int nHead,
        int nHeadKv,
        int nLayer,
        int nCtx,
        float rmsEps,
        float ropeBase,
        int ropeDim,
        RopeScalingType ropeScaling,
        float ropeScaleFactor,
        int ropeOrigCtx,
        float yarnExtFactor,
        float yarnAttnFactor) {

    public int headDim() {
        return nEmbd / nHead;
    }

    public RopeConfig toRopeConfig() {
        float freqScale = 1.0f;
        if (ropeScaling == RopeScalingType.LINEAR && ropeScaleFactor > 0.0f) {
            freqScale = 1.0f / ropeScaleFactor;
        }
        if (ropeScaling == RopeScalingType.YARN && ropeScaleFactor > 0.0f) {
            freqScale = 1.0f / ropeScaleFactor;
        }
        int hd = headDim();
        int effectiveRopeDim = ropeDim > 0 ? Math.min(ropeDim, hd) : hd;
        if (effectiveRopeDim % 2 != 0) {
            effectiveRopeDim--;
        }
        if (effectiveRopeDim <= 0) {
            effectiveRopeDim = hd;
        }
        float ext = ropeScaling == RopeScalingType.YARN ? (yarnExtFactor != 0.0f ? yarnExtFactor : 1.0f) : 0.0f;
        int origCtx = ropeOrigCtx > 0 ? ropeOrigCtx : nCtx;
        return new RopeConfig(
                ropeBase,
                freqScale,
                effectiveRopeDim,
                ropeScaling,
                ext,
                yarnAttnFactor > 0.0f ? yarnAttnFactor : 1.0f,
                32.0f,
                1.0f,
                origCtx);
    }

    public static ChatConfig fromGguf(GgufHeader header) {
        int nEmbd = requireInt(header, "llama.embedding_length");
        int nHead = requireInt(header, "llama.attention.head_count");
        int nHeadKv = intKv(header, "llama.attention.head_count_kv", nHead);
        int nLayer = requireInt(header, "llama.block_count");
        int nCtx = intKv(header, "llama.context_length", 512);
        int nVocab = intKv(header, "llama.vocab_size", guessVocab(header));
        float rmsEps = floatKv(header, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        float ropeBase = floatKv(header, "llama.rope.freq_base", 10000.0f);
        int ropeDim = intKv(header, "llama.rope.dimension_count", nEmbd / nHead);
        RopeScalingType scaling = parseScaling(stringKv(header, "llama.rope.scaling.type", "none"));
        float scaleFactor = floatKv(header, "llama.rope.scaling.factor", 0.0f);
        int origCtx = intKv(header, "llama.rope.scaling.original_context_length", nCtx);
        float yarnExt = floatKv(header, "llama.rope.scaling.yarn_ext_factor", 1.0f);
        float yarnAttn = floatKv(header, "llama.rope.scaling.attn_factor", 1.0f);
        return new ChatConfig(
                nVocab, nEmbd, nHead, nHeadKv, nLayer, nCtx, rmsEps, ropeBase,
                ropeDim, scaling, scaleFactor, origCtx, yarnExt, yarnAttn);
    }

    private static RopeScalingType parseScaling(String value) {
        if ("yarn".equalsIgnoreCase(value)) {
            return RopeScalingType.YARN;
        }
        if ("linear".equalsIgnoreCase(value)) {
            return RopeScalingType.LINEAR;
        }
        return RopeScalingType.NONE;
    }

    private static int guessVocab(GgufHeader header) {
        if (header.findTensor("token_embd.weight") != null) {
            return (int) header.findTensor("token_embd.weight").shape().ne(1);
        }
        return 32000;
    }

    private static int requireInt(GgufHeader header, String key) {
        GgufKvEntry entry = header.findKv(key);
        if (entry == null) {
            throw new IllegalArgumentException("missing kv " + key);
        }
        return ((Number) entry.value()).intValue();
    }

    private static int intKv(GgufHeader header, String key, int defaultValue) {
        GgufKvEntry entry = header.findKv(key);
        if (entry == null) {
            return defaultValue;
        }
        return ((Number) entry.value()).intValue();
    }

    private static float floatKv(GgufHeader header, String key, float defaultValue) {
        GgufKvEntry entry = header.findKv(key);
        if (entry == null) {
            return defaultValue;
        }
        return ((Number) entry.value()).floatValue();
    }

    private static String stringKv(GgufHeader header, String key, String defaultValue) {
        GgufKvEntry entry = header.findKv(key);
        if (entry == null) {
            return defaultValue;
        }
        return (String) entry.value();
    }
}
