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
import java.util.Random;
import java.util.Set;

/** Builds tiny llama-style GGUF fixtures for chat inference tests and demos. */
public final class MiniChatGgufBuilder {

    /** Smoke-test fixtures ({@code buildIdentityModel}, etc.). */
    public static final int N_EMBD = 32;
    public static final int N_HEAD = 4;
    public static final int N_HEAD_KV = 2;
    public static final int N_LAYER = 1;
    public static final int N_VOCAB = 4;
    public static final int N_CTX = 8;

    /**
     * Level-12 chat demo scale (tinygrad {@code apps/llm.py} catalog + llama3 chat tokens).
     */
    public static final class ChatDemo {
        public static final int N_EMBD = 768;
        public static final int N_HEAD = 12;
        public static final int N_HEAD_KV = 6;
        public static final int N_LAYER = 12;
        public static final int N_CTX = 2048;
        public static final int ROPE_DIM = 64;
        /** Full llama3.2:1b vocab ({@link ChatDemoVocab#LLAMA32_FULL_VOCAB}). */
        public static final int FULL_VOCAB = ChatDemoVocab.LLAMA32_FULL_VOCAB;
        /** Minimum pruned slice size when {@code TENSOR4J_CHAT_VOCAB=pruned}. */
        public static final int PRUNED_MIN_VOCAB = 512;
        public static final int TURN_COUNT = 4;

        private ChatDemo() {
        }
    }

    /** Reproducible init for {@link #buildOpenChatDemoModel()} (tinygrad real-GGUF path, not chain routing). */
    public static final long OPEN_CHAT_DEMO_SEED = 42L;

    private MiniChatGgufBuilder() {
    }

    public static GgufFile buildIdentityModel() {
        return buildWithShape(ModelShape.smoke(), false, false, "llama-spm", new String[] {"<s>", "a", "b", "</s>"}, false);
    }

    public static GgufFile buildTwoLayerModel() {
        return buildWithShape(
                new ModelShape(N_EMBD, N_HEAD, N_HEAD_KV, 2, N_CTX, 4),
                false,
                false,
                "llama-spm",
                new String[] {"<s>", "a", "b", "</s>"},
                false);
    }

    public static GgufFile buildYarnModel() {
        return buildWithShape(ModelShape.smoke(), true, false, "llama-spm", new String[] {"<s>", "a", "b", "</s>"}, false);
    }

    public static GgufFile buildQ4Model() {
        return buildWithShape(ModelShape.smoke(), false, true, "llama-spm", new String[] {"<s>", "a", "b", "</s>"}, false);
    }

    /** Llama3 BPE tokenizer + identity weights for encode/forward smoke. */
    public static GgufFile buildLlama3BpeModel() {
        return buildWithShape(
                ModelShape.smoke(),
                false,
                false,
                "llama3",
                new String[] {"<s>", "Hello", "a", "b", "</s>"},
                true);
    }

    /**
     * Llama3-style chat header tokens for {@link ChatTemplate#LLAMA3} parity (literal llama-spm vocab).
     */
    public static GgufFile buildLlama3TemplateModel() {
        String headerStart = "<|" + "start_header_id" + "|>";
        String headerEnd = "<|" + "end_header_id" + "|>";
        return buildWithShape(
                ModelShape.smoke(),
                false,
                false,
                "llama-spm",
                new String[] {
                    "<s>", "Hello", "</s>", headerStart, "user", "assistant", headerEnd, "\n", "\n\n", "<|eot_id|>"
                },
                true);
    }

    /** Same weights as {@link #buildLlama3BpeModel()} but omits {@code output.weight} (tied lm_head). */
    public static GgufFile buildTiedLmHeadModel() {
        return buildWithShape(
                ModelShape.smoke(),
                false,
                false,
                "llama3",
                new String[] {"<s>", "Hello", "a", "b", "</s>"},
                true,
                null,
                null,
                false);
    }

    /** Level-12 chain-routing demo — requires pruned vocab ({@code nEmbd >= nVocab}). */
    public static GgufFile buildChatDemoModel() {
        ChatDemoVocab.InferenceVocab vocab = ChatDemoVocab.load(ChatDemoVocabMode.PRUNED);
        if (vocab.vocabSize() > ModelShape.chatDemo().nEmbd()) {
            throw new IllegalStateException(
                    "chain demo requires pruned vocab with nVocab <= nEmbd (set TENSOR4J_CHAT_VOCAB=pruned "
                            + "and run capture --mode pruned)");
        }
        ModelShape shape = ModelShape.chatDemo();
        float[] emb = oneHotEmbeddingTable(vocab.vocabSize(), shape.nEmbd());
        float[] lmHead = chainLmHead(vocab.vocabSize(), shape.nEmbd());
        return buildChatDemoWithShape(shape, false, false, vocab, emb, lmHead, true, null, true);
    }

    /**
     * Level-12 open-ended demo: seeded weights, tied lm_head, llama3.2 BPE vocab
     * (full by default via {@link ChatDemoVocab#load()}).
     */
    public static GgufFile buildOpenChatDemoModel() {
        return buildOpenChatDemoModel(ChatDemoVocab.load());
    }

    /** Public for testability — explicit vocab mode (full vs pruned). */
    public static GgufFile buildOpenChatDemoModel(ChatDemoVocab.InferenceVocab vocab) {
        return buildChatDemoWithShape(
                ModelShape.chatDemo(),
                false,
                false,
                vocab,
                null,
                null,
                false,
                OPEN_CHAT_DEMO_SEED,
                false);
    }

    private static GgufFile buildChatDemoWithShape(
            ModelShape shape,
            boolean yarn,
            boolean q4Weights,
            ChatDemoVocab.InferenceVocab vocab,
            float[] embeddingOverride,
            float[] lmHeadOverride,
            boolean includeOutputWeight,
            Long weightSeed,
            boolean zeroAttentionOutput) {
        return buildWithShape(
                shape,
                yarn,
                q4Weights,
                vocab.pre(),
                vocab.tokens(),
                vocab.merges(),
                vocab.tokenTypes(),
                vocab.ignoreMerges(),
                vocab.bosTokenId(),
                vocab.eosTokenId(),
                embeddingOverride,
                lmHeadOverride,
                includeOutputWeight,
                weightSeed,
                zeroAttentionOutput);
    }

    public static GgufFile buildModel(int nLayer, boolean yarn, boolean q4Weights) {
        return buildWithShape(
                new ModelShape(N_EMBD, N_HEAD, N_HEAD_KV, nLayer, N_CTX, 4),
                yarn,
                q4Weights,
                "llama-spm",
                new String[] {"<s>", "a", "b", "</s>"},
                false);
    }

    public static GgufFile buildModel(
            int nLayer,
            boolean yarn,
            boolean q4Weights,
            String pre,
            String[] tokens,
            boolean ignoreMerges) {
        return buildWithShape(
                new ModelShape(N_EMBD, N_HEAD, N_HEAD_KV, nLayer, N_CTX, 4),
                yarn,
                q4Weights,
                pre,
                tokens,
                ignoreMerges,
                null,
                null);
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
        return buildWithShape(
                new ModelShape(N_EMBD, N_HEAD, N_HEAD_KV, nLayer, N_CTX, 4),
                yarn,
                q4Weights,
                pre,
                tokens,
                ignoreMerges,
                embeddingOverride,
                lmHeadOverride);
    }

    private static GgufFile buildWithShape(
            ModelShape shape,
            boolean yarn,
            boolean q4Weights,
            String pre,
            String[] tokens,
            String[] merges,
            int[] tokenTypes,
            boolean ignoreMerges,
            int bosId,
            int eosId,
            float[] embeddingOverride,
            float[] lmHeadOverride,
            boolean includeOutputWeight,
            Long weightSeed,
            boolean zeroAttentionOutput) {
        int nVocab = tokens.length;
        List<GgufKvEntry> kv = new ArrayList<>();
        kv.add(new GgufKvEntry("general.architecture", GgufType.STRING, "llama"));
        kv.add(new GgufKvEntry("llama.embedding_length", GgufType.UINT32, shape.nEmbd()));
        kv.add(new GgufKvEntry("llama.attention.head_count", GgufType.UINT32, shape.nHead()));
        kv.add(new GgufKvEntry("llama.attention.head_count_kv", GgufType.UINT32, shape.nHeadKv()));
        kv.add(new GgufKvEntry("llama.block_count", GgufType.UINT32, shape.nLayer()));
        kv.add(new GgufKvEntry("llama.context_length", GgufType.UINT32, shape.nCtx()));
        kv.add(new GgufKvEntry("llama.vocab_size", GgufType.UINT32, nVocab));
        kv.add(new GgufKvEntry("llama.rope.freq_base", GgufType.FLOAT32, 10000.0f));
        kv.add(new GgufKvEntry("llama.rope.dimension_count", GgufType.UINT32, shape.ropeDim()));
        kv.add(new GgufKvEntry("llama.attention.layer_norm_rms_epsilon", GgufType.FLOAT32, 1e-5f));
        if (yarn) {
            kv.add(new GgufKvEntry("llama.rope.scaling.type", GgufType.STRING, "yarn"));
            kv.add(new GgufKvEntry("llama.rope.scaling.factor", GgufType.FLOAT32, 2.0f));
            kv.add(new GgufKvEntry("llama.rope.scaling.original_context_length", GgufType.UINT32, 4));
            kv.add(new GgufKvEntry("llama.rope.scaling.yarn_ext_factor", GgufType.FLOAT32, 1.0f));
            kv.add(new GgufKvEntry("llama.rope.scaling.attn_factor", GgufType.FLOAT32, 1.0f));
        }
        addTokenizerKv(kv, pre, tokens, merges, tokenTypes, ignoreMerges, bosId, eosId);

        int nEmbd = shape.nEmbd();
        List<GgufTensorPayload> tensors = new ArrayList<>();
        GgmlType matType = q4Weights ? GgmlType.Q4_0 : GgmlType.F32;
        Random rng = weightSeed == null ? null : new Random(weightSeed);
        float[] emb = embeddingOverride == null
                ? (rng == null ? embeddingTable(nVocab, nEmbd) : seededEmbeddingTable(nVocab, nEmbd, rng))
                : embeddingOverride;
        float[] lmHead = lmHeadOverride == null ? identityMat(nEmbd, nVocab) : lmHeadOverride;
        tensors.add(weight("token_embd.weight", matType, GgmlTensorShape.of(nEmbd, nVocab), emb));
        tensors.add(f32("output_norm.weight", GgmlTensorShape.of(nEmbd), ones(nEmbd)));
        if (includeOutputWeight) {
            tensors.add(weight("output.weight", matType, GgmlTensorShape.of(nEmbd, nVocab), lmHead));
        }

        float attnScale = rng == null ? 0f : 0.02f / (float) Math.sqrt(nEmbd);
        float ffnScale = rng == null ? 0f : 0.01f / (float) Math.sqrt(nEmbd);
        int kvWidth = shape.nHeadKv() * (nEmbd / shape.nHead());
        for (int layer = 0; layer < shape.nLayer(); layer++) {
            String prefix = "blk." + layer + ".";
            tensors.add(f32(prefix + "attn_norm.weight", GgmlTensorShape.of(nEmbd), ones(nEmbd)));
            tensors.add(weight(prefix + "attn_q.weight", matType, GgmlTensorShape.of(nEmbd, nEmbd),
                    rng == null ? identityMat(nEmbd, nEmbd) : seededMat(nEmbd, nEmbd, rng, attnScale)));
            tensors.add(weight(prefix + "attn_k.weight", matType, GgmlTensorShape.of(nEmbd, kvWidth),
                    rng == null ? identityMat(nEmbd, kvWidth) : seededMat(nEmbd, kvWidth, rng, attnScale)));
            tensors.add(weight(prefix + "attn_v.weight", matType, GgmlTensorShape.of(nEmbd, kvWidth),
                    rng == null ? identityMat(nEmbd, kvWidth) : seededMat(nEmbd, kvWidth, rng, attnScale)));
            tensors.add(weight(prefix + "attn_output.weight", matType, GgmlTensorShape.of(nEmbd, nEmbd),
                    woMatrix(nEmbd, rng, attnScale, zeroAttentionOutput)));
            tensors.add(f32(prefix + "ffn_norm.weight", GgmlTensorShape.of(nEmbd), ones(nEmbd)));
            tensors.add(f32(prefix + "ffn_gate.weight", GgmlTensorShape.of(nEmbd, nEmbd),
                    rng == null ? zeroMat(nEmbd, nEmbd) : seededMat(nEmbd, nEmbd, rng, ffnScale)));
            tensors.add(weight(prefix + "ffn_up.weight", matType, GgmlTensorShape.of(nEmbd, nEmbd),
                    rng == null ? identityMat(nEmbd, nEmbd) : seededMat(nEmbd, nEmbd, rng, ffnScale)));
            tensors.add(f32(prefix + "ffn_down.weight", GgmlTensorShape.of(nEmbd, nEmbd),
                    rng == null ? zeroMat(nEmbd, nEmbd) : seededMat(nEmbd, nEmbd, rng, ffnScale)));
        }

        byte[] bytes = GgufWriter.writeFile(GgufConstants.VERSION, kv, tensors);
        return GgufReader.readFile(bytes);
    }

    private static GgufFile buildWithShape(
            ModelShape shape,
            boolean yarn,
            boolean q4Weights,
            String pre,
            String[] tokens,
            boolean ignoreMerges) {
        return buildWithShape(
                shape, yarn, q4Weights, pre, tokens, new String[0], null, ignoreMerges, 0, tokens.length - 1,
                null, null, true, null, false);
    }

    private static GgufFile buildWithShape(
            ModelShape shape,
            boolean yarn,
            boolean q4Weights,
            String pre,
            String[] tokens,
            boolean ignoreMerges,
            float[] embeddingOverride,
            float[] lmHeadOverride) {
        return buildWithShape(
                shape, yarn, q4Weights, pre, tokens, new String[0], null, ignoreMerges, 0, tokens.length - 1,
                embeddingOverride, lmHeadOverride, true, null, false);
    }

    private static GgufFile buildWithShape(
            ModelShape shape,
            boolean yarn,
            boolean q4Weights,
            String pre,
            String[] tokens,
            boolean ignoreMerges,
            float[] embeddingOverride,
            float[] lmHeadOverride,
            boolean includeOutputWeight) {
        return buildWithShape(
                shape, yarn, q4Weights, pre, tokens, new String[0], null, ignoreMerges, 0, tokens.length - 1,
                embeddingOverride, lmHeadOverride, includeOutputWeight, null, false);
    }

    private static GgufFile buildWithShape(
            ModelShape shape,
            boolean yarn,
            boolean q4Weights,
            String pre,
            String[] tokens,
            boolean ignoreMerges,
            float[] embeddingOverride,
            float[] lmHeadOverride,
            boolean includeOutputWeight,
            Long weightSeed) {
        return buildWithShape(
                shape, yarn, q4Weights, pre, tokens, new String[0], null, ignoreMerges, 0, tokens.length - 1,
                embeddingOverride, lmHeadOverride, includeOutputWeight, weightSeed, false);
    }

    private record ModelShape(int nEmbd, int nHead, int nHeadKv, int nLayer, int nCtx, int ropeDim) {
        static ModelShape smoke() {
            return new ModelShape(N_EMBD, N_HEAD, N_HEAD_KV, N_LAYER, N_CTX, 4);
        }

        static ModelShape chatDemo() {
            return new ModelShape(
                    ChatDemo.N_EMBD,
                    ChatDemo.N_HEAD,
                    ChatDemo.N_HEAD_KV,
                    ChatDemo.N_LAYER,
                    ChatDemo.N_CTX,
                    ChatDemo.ROPE_DIM);
        }
    }

    private static void addTokenizerKv(List<GgufKvEntry> kv, String pre, String[] tokens, boolean ignoreMerges) {
        addTokenizerKv(kv, pre, tokens, new String[0], null, ignoreMerges, 0, tokens.length - 1);
    }

    private static void addTokenizerKv(
            List<GgufKvEntry> kv,
            String pre,
            String[] tokens,
            String[] merges,
            int[] tokenTypes,
            boolean ignoreMerges,
            int bosId,
            int eosId) {
        kv.add(new GgufKvEntry("tokenizer.ggml.model", GgufType.STRING, "llama"));
        kv.add(new GgufKvEntry("tokenizer.ggml.pre", GgufType.STRING, pre));
        kv.add(new GgufKvEntry("tokenizer.ggml.tokens", GgufType.ARRAY,
                new GgufArrayValue(GgufType.STRING, tokens)));
        kv.add(new GgufKvEntry("tokenizer.ggml.merges", GgufType.ARRAY,
                new GgufArrayValue(GgufType.STRING, merges == null ? new String[0] : merges)));
        int[] types = tokenTypes == null ? defaultTokenTypes(tokens.length) : tokenTypes;
        Object[] typeObjs = new Object[types.length];
        for (int i = 0; i < types.length; i++) {
            typeObjs[i] = types[i];
        }
        kv.add(new GgufKvEntry("tokenizer.ggml.token_type", GgufType.ARRAY,
                new GgufArrayValue(GgufType.INT32, typeObjs)));
        kv.add(new GgufKvEntry("tokenizer.ggml.bos_token_id", GgufType.UINT32, bosId));
        kv.add(new GgufKvEntry("tokenizer.ggml.eos_token_id", GgufType.UINT32, eosId));
        if (ignoreMerges) {
            kv.add(new GgufKvEntry("tokenizer.ggml.ignore_merges", GgufType.BOOL, true));
        }
    }

    private static int[] defaultTokenTypes(int n) {
        int[] types = new int[n];
        for (int i = 0; i < n; i++) {
            types[i] = (i == 0 || i == n - 1) ? 3 : 1;
        }
        return types;
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

    private static float[] embeddingTable(int nVocab, int nEmbd) {
        float[] table = new float[nVocab * nEmbd];
        for (int v = 0; v < nVocab; v++) {
            for (int d = 0; d < nEmbd; d++) {
                table[d + v * nEmbd] = (v + 1) * 0.1f + d * 0.01f;
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

    /** Chain routing needs embeddings unchanged — zero {@code attn_output} (FFN already no-op). */
    private static float[] woMatrix(int nEmbd, Random rng, float attnScale, boolean zeroAttentionOutput) {
        if (zeroAttentionOutput) {
            return zeroMat(nEmbd, nEmbd);
        }
        return rng == null ? identityMat(nEmbd, nEmbd) : seededMat(nEmbd, nEmbd, rng, attnScale);
    }

    private static float[] identityMat(int rows, int cols) {
        float[] out = new float[rows * cols];
        int n = Math.min(rows, cols);
        for (int i = 0; i < n; i++) {
            out[ggmlIndex(rows, i, i)] = 1.0f;
        }
        return out;
    }

    /** ggml row-major: {@code ne[0]} stride is 1, {@code ne[1]} stride is {@code ne0}. */
    private static int ggmlIndex(int ne0, int d0, int d1) {
        return d0 + d1 * ne0;
    }

    /** One-hot rows so lm_head can route token i → i+1 deterministically. */
    private static float[] oneHotEmbeddingTable(int nVocab, int nEmbd) {
        if (nVocab > nEmbd) {
            throw new IllegalArgumentException("one-hot chat demo requires nEmbd >= nVocab");
        }
        float[] table = new float[nVocab * nEmbd];
        for (int v = 0; v < nVocab; v++) {
            table[v * nEmbd + v] = 1.0f;
        }
        return table;
    }

    /** Sparse lm_head: after token id {@code i}, boost logit for {@code i + 1}. */
    private static float[] chainLmHead(int nVocab, int nEmbd) {
        float[] lmHead = new float[nEmbd * nVocab];
        float weight = 36.0f;
        for (int src = 1; src < nVocab - 1; src++) {
            int dst = src + 1;
            lmHead[src + dst * nEmbd] = weight;
        }
        return lmHead;
    }

    private static float[] seededEmbeddingTable(int nVocab, int nEmbd, Random rng) {
        float scale = 0.02f / (float) Math.sqrt(nEmbd);
        float[] table = new float[nVocab * nEmbd];
        for (int i = 0; i < table.length; i++) {
            table[i] = (float) (rng.nextGaussian() * scale);
        }
        return table;
    }

    private static float[] seededMat(int rows, int cols, Random rng, float scale) {
        float[] out = new float[rows * cols];
        for (int d1 = 0; d1 < cols; d1++) {
            for (int d0 = 0; d0 < rows; d0++) {
                out[ggmlIndex(rows, d0, d1)] = (float) (rng.nextGaussian() * scale);
            }
        }
        return out;
    }

    private static byte[] f32Bytes(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            buffer.putFloat(value);
        }
        return buffer.array();
    }
}
