/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.fixture;

import com.github.tensor4j.models.chat.ChatTokenizer;
import com.github.tensor4j.runtime.ggml.GgmlQuant;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import com.github.tensor4j.runtime.gguf.GgufArrayValue;
import com.github.tensor4j.runtime.gguf.GgufConstants;
import com.github.tensor4j.runtime.gguf.GgufFile;
import com.github.tensor4j.runtime.gguf.GgufKvEntry;
import com.github.tensor4j.runtime.gguf.GgufReader;
import com.github.tensor4j.runtime.gguf.GgufTensorPayload;
import com.github.tensor4j.runtime.gguf.GgufType;
import com.github.tensor4j.runtime.gguf.GgufWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/** Builds tiny llama-style GGUF fixtures for chat inference tests and demos. */
public final class MiniChatGgufBuilder {

    public static final int N_EMBD = 32;
    public static final int N_HEAD = 4;
    public static final int N_HEAD_KV = 2;
    public static final int N_LAYER = 1;
    public static final int N_VOCAB = 4;
    public static final int N_CTX = 8;

    private MiniChatGgufBuilder() {
    }

    public static GgufFile buildIdentityModel() {
        return buildModel(N_LAYER, false, false);
    }

    public static GgufFile buildTwoLayerModel() {
        return buildModel(2, false, false);
    }

    public static GgufFile buildYarnModel() {
        return buildModel(N_LAYER, true, false);
    }

    public static GgufFile buildQ4Model() {
        return buildModel(N_LAYER, false, true);
    }

    /** Llama3 BPE tokenizer + identity weights for encode/forward smoke. */
    public static GgufFile buildLlama3BpeModel() {
        return buildModel(N_LAYER, false, false, "llama3", new String[] {"<s>", "Hello", "a", "b", "</s>"}, true);
    }

    /**
     * Llama3 BPE + lm_head tuned so {@code Hello} completes to {@code  there!} under quality sampling.
     */
    public static GgufFile buildChatDemoModel() {
        String[] tokens = llama3ChatDemoTokens(
                "<s>", "Hello", " there", "!", " How", " can", " I", " help", "?", "</s>");
        float[] emb = oneHotEmbeddingTable(tokens.length);
        float[] lmHead = chainLmHead(tokens.length);
        return buildModel(N_LAYER, false, false, "llama3", tokens, true, emb, lmHead);
    }

    private static String[] llama3ChatDemoTokens(String... pieces) {
        String[] out = new String[pieces.length];
        for (int i = 0; i < pieces.length; i++) {
            String piece = pieces[i];
            if ("<s>".equals(piece) || "</s>".equals(piece)) {
                out[i] = piece;
            } else {
                out[i] = ChatTokenizer.llama3VocabPiece(piece);
            }
        }
        return out;
    }

    public static GgufFile buildModel(int nLayer, boolean yarn, boolean q4Weights) {
        return buildModel(nLayer, yarn, q4Weights, "llama-spm", new String[] {"<s>", "a", "b", "</s>"}, false);
    }

    public static GgufFile buildModel(
            int nLayer,
            boolean yarn,
            boolean q4Weights,
            String pre,
            String[] tokens,
            boolean ignoreMerges) {
        return buildModel(nLayer, yarn, q4Weights, pre, tokens, ignoreMerges, null, null);
    }

    public static GgufFile buildModel(
            int nLayer,
            boolean yarn,
            boolean q4Weights,
            String pre,
            String[] tokens,
            boolean ignoreMerges,
            float[] embeddingOverride,
            float[] lmHeadOverride) {
        int nVocab = tokens.length;
        List<GgufKvEntry> kv = new ArrayList<>();
        kv.add(new GgufKvEntry("general.architecture", GgufType.STRING, "llama"));
        kv.add(new GgufKvEntry("llama.embedding_length", GgufType.UINT32, N_EMBD));
        kv.add(new GgufKvEntry("llama.attention.head_count", GgufType.UINT32, N_HEAD));
        kv.add(new GgufKvEntry("llama.attention.head_count_kv", GgufType.UINT32, N_HEAD_KV));
        kv.add(new GgufKvEntry("llama.block_count", GgufType.UINT32, nLayer));
        kv.add(new GgufKvEntry("llama.context_length", GgufType.UINT32, N_CTX));
        kv.add(new GgufKvEntry("llama.vocab_size", GgufType.UINT32, nVocab));
        kv.add(new GgufKvEntry("llama.rope.freq_base", GgufType.FLOAT32, 10000.0f));
        kv.add(new GgufKvEntry("llama.rope.dimension_count", GgufType.UINT32, 4));
        kv.add(new GgufKvEntry("llama.attention.layer_norm_rms_epsilon", GgufType.FLOAT32, 1e-5f));
        if (yarn) {
            kv.add(new GgufKvEntry("llama.rope.scaling.type", GgufType.STRING, "yarn"));
            kv.add(new GgufKvEntry("llama.rope.scaling.factor", GgufType.FLOAT32, 2.0f));
            kv.add(new GgufKvEntry("llama.rope.scaling.original_context_length", GgufType.UINT32, 4));
            kv.add(new GgufKvEntry("llama.rope.scaling.yarn_ext_factor", GgufType.FLOAT32, 1.0f));
            kv.add(new GgufKvEntry("llama.rope.scaling.attn_factor", GgufType.FLOAT32, 1.0f));
        }
        addTokenizerKv(kv, pre, tokens, ignoreMerges);

        List<GgufTensorPayload> tensors = new ArrayList<>();
        GgmlType matType = q4Weights ? GgmlType.Q4_0 : GgmlType.F32;
        float[] emb = embeddingOverride == null ? embeddingTable(nVocab) : embeddingOverride;
        float[] lmHead = lmHeadOverride == null ? identityMat(N_EMBD, nVocab) : lmHeadOverride;
        tensors.add(weight("token_embd.weight", matType, GgmlTensorShape.of(N_EMBD, nVocab), emb));
        tensors.add(f32("output_norm.weight", GgmlTensorShape.of(N_EMBD), ones(N_EMBD)));
        tensors.add(weight("output.weight", matType, GgmlTensorShape.of(N_EMBD, nVocab), lmHead));

        int kvWidth = N_HEAD_KV * (N_EMBD / N_HEAD);
        for (int layer = 0; layer < nLayer; layer++) {
            String prefix = "blk." + layer + ".";
            tensors.add(f32(prefix + "attn_norm.weight", GgmlTensorShape.of(N_EMBD), ones(N_EMBD)));
            tensors.add(weight(prefix + "attn_q.weight", matType, GgmlTensorShape.of(N_EMBD, N_EMBD),
                    identityMat(N_EMBD, N_EMBD)));
            tensors.add(weight(prefix + "attn_k.weight", matType, GgmlTensorShape.of(N_EMBD, kvWidth),
                    identityMat(N_EMBD, kvWidth)));
            tensors.add(weight(prefix + "attn_v.weight", matType, GgmlTensorShape.of(N_EMBD, kvWidth),
                    identityMat(N_EMBD, kvWidth)));
            tensors.add(weight(prefix + "attn_output.weight", matType, GgmlTensorShape.of(N_EMBD, N_EMBD),
                    identityMat(N_EMBD, N_EMBD)));
            tensors.add(f32(prefix + "ffn_norm.weight", GgmlTensorShape.of(N_EMBD), ones(N_EMBD)));
            tensors.add(f32(prefix + "ffn_gate.weight", GgmlTensorShape.of(N_EMBD, N_EMBD), zeroMat(N_EMBD, N_EMBD)));
            tensors.add(weight(prefix + "ffn_up.weight", matType, GgmlTensorShape.of(N_EMBD, N_EMBD),
                    identityMat(N_EMBD, N_EMBD)));
            tensors.add(f32(prefix + "ffn_down.weight", GgmlTensorShape.of(N_EMBD, N_EMBD), zeroMat(N_EMBD, N_EMBD)));
        }

        byte[] bytes = GgufWriter.writeFile(GgufConstants.VERSION, kv, tensors);
        return GgufReader.readFile(bytes);
    }

    private static void addTokenizerKv(List<GgufKvEntry> kv, String pre, String[] tokens, boolean ignoreMerges) {
        kv.add(new GgufKvEntry("tokenizer.ggml.model", GgufType.STRING, "llama"));
        kv.add(new GgufKvEntry("tokenizer.ggml.pre", GgufType.STRING, pre));
        kv.add(new GgufKvEntry("tokenizer.ggml.tokens", GgufType.ARRAY,
                new GgufArrayValue(GgufType.STRING, tokens)));
        kv.add(new GgufKvEntry("tokenizer.ggml.merges", GgufType.ARRAY,
                new GgufArrayValue(GgufType.STRING, new String[0])));
        int[] types = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            types[i] = (i == 0 || i == tokens.length - 1) ? 3 : 1;
        }
        Object[] typeObjs = new Object[types.length];
        for (int i = 0; i < types.length; i++) {
            typeObjs[i] = types[i];
        }
        kv.add(new GgufKvEntry("tokenizer.ggml.token_type", GgufType.ARRAY,
                new GgufArrayValue(GgufType.INT32, typeObjs)));
        kv.add(new GgufKvEntry("tokenizer.ggml.bos_token_id", GgufType.UINT32, 0));
        kv.add(new GgufKvEntry("tokenizer.ggml.eos_token_id", GgufType.UINT32, tokens.length - 1));
        if (ignoreMerges) {
            kv.add(new GgufKvEntry("tokenizer.ggml.ignore_merges", GgufType.BOOL, true));
        }
    }

    private static GgufTensorPayload weight(String name, GgmlType type, GgmlTensorShape shape, float[] data) {
        if (type == GgmlType.F32) {
            return f32(name, shape, data);
        }
        byte[] q4 = GgmlQuant.quantizeRowQ4_0Reference(padToBlock(data));
        return new GgufTensorPayload(name, GgmlType.Q4_0, shape, q4);
    }

    private static float[] padToBlock(float[] data) {
        if (data.length % GgmlQuant.QK4_0 == 0) {
            return data;
        }
        int padded = ((data.length + GgmlQuant.QK4_0 - 1) / GgmlQuant.QK4_0) * GgmlQuant.QK4_0;
        float[] out = new float[padded];
        System.arraycopy(data, 0, out, 0, data.length);
        return out;
    }

    private static GgufTensorPayload f32(String name, GgmlTensorShape shape, float[] data) {
        return new GgufTensorPayload(name, GgmlType.F32, shape, f32Bytes(data));
    }

    private static float[] embeddingTable(int nVocab) {
        float[] table = new float[nVocab * N_EMBD];
        for (int v = 0; v < nVocab; v++) {
            for (int d = 0; d < N_EMBD; d++) {
                table[d + v * N_EMBD] = (v + 1) * 0.1f + d * 0.01f;
            }
        }
        return table;
    }

    private static float[] ones(int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = 1.0f;
        }
        return out;
    }

    private static float[] zeroMat(int rows, int cols) {
        return new float[rows * cols];
    }

    private static float[] identityMat(int rows, int cols) {
        float[] out = new float[rows * cols];
        int n = Math.min(rows, cols);
        for (int i = 0; i < n; i++) {
            out[i * cols + i] = 1.0f;
        }
        return out;
    }

    /** One-hot rows so lm_head can route token i → i+1 deterministically. */
    private static float[] oneHotEmbeddingTable(int nVocab) {
        float[] table = new float[nVocab * N_EMBD];
        for (int v = 0; v < nVocab && v < N_EMBD; v++) {
            table[v * N_EMBD + v] = 1.0f;
        }
        return table;
    }

    /** Sparse lm_head: after token id {@code i}, boost logit for {@code i + 1}. */
    private static float[] chainLmHead(int nVocab) {
        float[] lmHead = new float[N_EMBD * nVocab];
        float weight = 12.0f;
        for (int src = 1; src < nVocab - 1; src++) {
            int dst = src + 1;
            if (src < N_EMBD) {
                lmHead[src * nVocab + dst] = weight;
            }
        }
        return lmHead;
    }

    private static byte[] f32Bytes(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            buffer.putFloat(value);
        }
        return buffer.array();
    }
}
