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

/**
 * Llama 3.2 chat inference vocab for level-12 fixtures (tinygrad {@code apps/llm.py}).
 *
 * <p>Default: full tokenizer ({@code tinygrad-llama32-chat-vocab-full.json}, gitignored — regenerate
 * via {@code python tools/capture_tinygrad_chat_vocab.py --mode full}). Optional pruned slice:
 * {@code TENSOR4J_CHAT_VOCAB=pruned} + {@code --mode pruned}.
 */
public final class ChatDemoVocab {

    /** Llama 3.2 1B instruct GGUF vocab size (tinygrad {@code llama3.2:1b}). */
    public static final int LLAMA32_FULL_VOCAB = 128256;

    static final String FULL_RESOURCE = "/fixtures/tinygrad-llama32-chat-vocab-full.json";
    static final String PRUNED_RESOURCE = "/fixtures/tinygrad-llama32-chat-vocab-pruned.json";

    /** Parsed tokenizer metadata + tables for GGUF fixture build. */
    public record InferenceVocab(
            ChatDemoVocabMode mode,
            String source,
            String pre,
            boolean ignoreMerges,
            int bosTokenId,
            int eosTokenId,
            String[] tokens,
            String[] merges,
            int[] tokenTypes,
            int fullVocabSize) {

        public int vocabSize() {
            return tokens.length;
        }

        public boolean isFull() {
            return mode == ChatDemoVocabMode.FULL;
        }
    }

    private static volatile InferenceVocab cached;
    private static volatile ChatDemoVocabMode cachedMode;

    private ChatDemoVocab() {
    }

    /** Uses {@link ChatDemoVocabMode#fromEnvironment()} — default {@link ChatDemoVocabMode#FULL}. */
    public static InferenceVocab load() {
        return load(ChatDemoVocabMode.fromEnvironment());
    }

    public static InferenceVocab load(ChatDemoVocabMode mode) {
        if (cached != null && cachedMode == mode) {
            return cached;
        }
        synchronized (ChatDemoVocab.class) {
            if (cached != null && cachedMode == mode) {
                return cached;
            }
            InferenceVocab loaded = parse(FixtureJson.readResource(resourcePath(mode)), mode);
            cached = loaded;
            cachedMode = mode;
            return loaded;
        }
    }

    /** Public for testability. */
    public static boolean resourcePresent(ChatDemoVocabMode mode) {
        return ChatDemoVocab.class.getResource(resourcePath(mode)) != null;
    }

    /** JUnit {@code @EnabledIf} — full llama3.2 fixture on classpath. */
    @SuppressWarnings("unused")
    public static boolean fullFixturePresent() {
        return resourcePresent(ChatDemoVocabMode.FULL);
    }

    /** JUnit {@code @EnabledIf} — pruned slice fixture on classpath. */
    @SuppressWarnings("unused")
    public static boolean prunedFixturePresent() {
        return resourcePresent(ChatDemoVocabMode.PRUNED);
    }

    /** Public for testability — reset singleton between mode tests. */
    public static void clearCacheForTests() {
        synchronized (ChatDemoVocab.class) {
            cached = null;
            cachedMode = null;
        }
    }

    static String resourcePath(ChatDemoVocabMode mode) {
        return mode == ChatDemoVocabMode.PRUNED ? PRUNED_RESOURCE : FULL_RESOURCE;
    }

    static InferenceVocab parse(String json, ChatDemoVocabMode mode) {
        String source = FixtureJson.stringField(json, "source");
        String pre = FixtureJson.stringField(json, "pre");
        boolean ignoreMerges = FixtureJson.booleanField(json, "ignore_merges", false);
        int bos = FixtureJson.intField(json, "bos_token_id", 0);
        int eos = FixtureJson.intField(json, "eos_token_id", -1);
        String[] tokens = FixtureJson.stringArrayField(json, "tokens");
        String[] merges = FixtureJson.stringArrayField(json, "merges");
        int[] tokenTypes = FixtureJson.intArrayField(json, "token_types");
        int fullVocab = FixtureJson.intField(json, "full_vocab_size", tokens.length);
        if (eos < 0) {
            eos = tokens.length - 1;
        }
        if (tokenTypes.length != tokens.length) {
            throw new IllegalStateException("token_types length mismatch");
        }
        return new InferenceVocab(mode, source, pre, ignoreMerges, bos, eos, tokens, merges, tokenTypes, fullVocab);
    }
}
