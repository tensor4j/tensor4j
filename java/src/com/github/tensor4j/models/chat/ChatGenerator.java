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

import com.github.tensor4j.models.chat.reference.TinygradGenerateReference;
import java.util.Arrays;
import java.util.function.Consumer;

/**
 * Stateful decode loop — chunked prefill, KV prefix reuse, streaming
 * (tinygrad {@code apps/llm.py} {@code generate()} + {@code get_start_pos()}).
 */
public final class ChatGenerator {

    private final ChatModel model;
    private final ChatGenerationOptions options;
    private final ChatTokenizer tokenizer;
    private final ChatSamplingRng samplingRng;
    private final ChatSamplerState samplerState;
    private int[] cachedTokens = new int[0];
    private int[] conversationTokens = new int[0];

    public ChatGenerator(ChatModel model, ChatGenerationOptions options) {
        this.model = model;
        this.options = options;
        this.tokenizer = model.tokenizer();
        this.samplingRng = ChatSamplingRng.fromOptions(options);
        this.samplerState = new ChatSamplerState(tokenizer.vocabSize());
    }

    /** Longest shared prefix between {@code tokens[:-1]} and {@code cached} (tinygrad {@code get_start_pos}). */
    public static int sharedPrefixLength(int[] tokens, int[] cached) {
        int limit = Math.min(tokens.length - 1, cached.length);
        int matched = 0;
        for (int i = 0; i < limit; i++) {
            if (tokens[i] != cached[i]) {
                break;
            }
            matched++;
        }
        return matched;
    }

    /**
     * Prefill slice lengths for {@code tokens[startPos:promptLen]} — same loop as
     * {@link #prefillChunked(ChatModel, int[], int, int)} and tinygrad {@code generate()} prefill.
     */
    static int[] prefillSliceLengths(int promptLen, int startPos, int chunkSize) {
        if (startPos >= promptLen) {
            throw new IllegalArgumentException("startPos beyond prompt length");
        }
        int chunk = Math.max(1, chunkSize);
        int remaining = promptLen - startPos;
        int count = (remaining + chunk - 1) / chunk;
        int[] slices = new int[count];
        int pos = startPos;
        for (int i = 0; i < count; i++) {
            int len = Math.min(chunk, promptLen - pos);
            slices[i] = len;
            pos += len;
        }
        return slices;
    }

    /**
     * Prefill vs decode forward flags for tinygrad {@code test_chunked_prefill}: every prompt-chunk
     * forward is prefill; each post-prompt token yield adds one single-token (decode) forward.
     */
    static boolean[] forwardPrefillFlags(int promptLen, int chunkSize, int yieldCount) {
        int[] slices = prefillSliceLengths(promptLen, 0, chunkSize);
        boolean[] flags = new boolean[slices.length + Math.max(0, yieldCount - 1)];
        Arrays.fill(flags, 0, slices.length, true);
        return flags;
    }

    /**
     * Session-scoped generate with KV prefix reuse — mirrors tinygrad {@code Transformer.generate()}
     * which always tracks {@code _cached_tokens}.
     */
    ChatGenerationResult generateWithKvReuse(int[] promptIds) {
        return generate(promptIds, true, null);
    }

    public void resetSession() {
        cachedTokens = new int[0];
        conversationTokens = new int[0];
        samplerState.reset();
        model.resetCache();
    }

    /** Single-turn generation — always fresh KV cache. */
    public ChatGenerationResult generate(String prompt, ChatTemplate template) {
        return generate(template.encodePromptForGeneration(tokenizer, prompt), false, null);
    }

    /** Single-turn from token ids. */
    public ChatGenerationResult generate(int[] promptIds) {
        return generate(promptIds, false, null);
    }

    /** Multi-turn: append user message to session history and reuse KV prefix when possible. */
    public ChatGenerationResult continueConversation(String userMessage, ChatTemplate template) {
        samplerState.reset();
        int[] promptIds = buildPromptIds(userMessage, template);
        ChatGenerationResult result = generate(promptIds, kvReuseEnabled(), null);
        finishTurn(promptIds, result.forwardedTokenIds(), template);
        return result;
    }

    /** Multi-turn with streaming token pieces (stdout-friendly). */
    public ChatGenerationResult continueConversationStreaming(
            String userMessage, ChatTemplate template, Consumer<String> onPiece) {
        samplerState.reset();
        int[] promptIds = buildPromptIds(userMessage, template);
        ChatGenerationResult result = generate(promptIds, kvReuseEnabled(), onPiece);
        finishTurn(promptIds, result.forwardedTokenIds(), template);
        return result;
    }

    /** When {@code TENSOR4J_CHAT_NO_KV_REUSE=true}, prefill the full prompt every turn (debug). */
    public static boolean kvReuseEnabled() {
        return !Boolean.parseBoolean(System.getenv().getOrDefault("TENSOR4J_CHAT_NO_KV_REUSE", "false"));
    }

    /** Token ids that will be prefilled on the next {@link #continueConversation} call. */
    public int[] planPromptIds(String userMessage, ChatTemplate template) {
        return buildPromptIds(userMessage, template);
    }

    /** Full session transcript token ids (after the last completed turn). */
    public int[] sessionTokenIds() {
        return conversationTokens.clone();
    }

    /** Cached prompt ids used for KV prefix reuse (tinygrad {@code _cached_tokens}). */
    public int[] cachedTokenIds() {
        return cachedTokens.clone();
    }

    public int kvLength() {
        return model.kvLength();
    }

    public ChatModel model() {
        return model;
    }

    public ChatGenerationOptions options() {
        return options;
    }

    private int[] buildPromptIds(String userMessage, ChatTemplate template) {
        int[] turn = concat(
                template.encodeUserTurn(tokenizer, userMessage),
                template.encodeAssistantPrime(tokenizer));
        return conversationTokens.length == 0
                ? concat(template.encodePrefix(tokenizer), turn)
                : concat(conversationTokens, turn);
    }

    /** Package-private: session token ids after multi-turn turns (tests). */
    int[] sessionTokenIdsForTests() {
        return sessionTokenIds();
    }

    /**
     * Append assistant {@code eot_id} to session + KV when the model stops without one
     * (tinygrad interactive chat expects a closed turn before the next user header).
     */
    private void finishTurn(int[] promptIds, int[] forwarded, ChatTemplate template) {
        int[] closed = closeAssistantTurn(forwarded, template);
        conversationTokens = concat(promptIds, closed);
        cachedTokens = conversationTokens;
        if (Boolean.parseBoolean(System.getenv().getOrDefault("TENSOR4J_CHAT_DEBUG", "false"))) {
            System.err.printf(
                    "chat session: %d tokens, ends with eot=%s%n",
                    conversationTokens.length,
                    conversationTokens.length > 0
                            && conversationTokens[conversationTokens.length - 1] == tokenizer.eosId());
        }
    }

    private int[] closeAssistantTurn(int[] forwarded, ChatTemplate template) {
        if (template != ChatTemplate.LLAMA3) {
            return forwarded;
        }
        if (forwarded.length > 0 && forwarded[forwarded.length - 1] == tokenizer.eosId()) {
            return forwarded;
        }
        int[] eot = template.encodeEndTurn(tokenizer);
        model.forward(eot);
        return concat(forwarded, eot);
    }

    private ChatGenerationResult generate(int[] promptIds, boolean reusePrefix, Consumer<String> onPiece) {
        if (promptIds.length == 0) {
            return new ChatGenerationResult("", 0, options.mode().name(), 0, new int[0], new int[0]);
        }

        int startPos = 0;
        if (reusePrefix && cachedTokens.length > 0) {
            startPos = sharedPrefixLength(promptIds, cachedTokens);
        }
        if (startPos == 0 || model.kvLength() != startPos) {
            model.resetCache();
            startPos = 0;
        }

        float[] logits = prefillChunked(promptIds, startPos, options.prefillChunkSize());

        StringBuilder completion = new StringBuilder();
        int[] generatedIds = new int[options.maxNewTokens()];
        int[] forwardedIds = new int[options.maxNewTokens()];
        int generated = 0;
        int forwarded = 0;
        for (int step = 0; step < options.maxNewTokens(); step++) {
            int next = ChatSampler.sample(logits, options, generated, samplerState, samplingRng);
            if (TinygradGenerateReference.shouldStop(next, tokenizer.eosId(), generated, options.minNewTokens())) {
                // tinygrad generate() appends eos to ids / _cached_tokens before stopping
                forwardedIds[forwarded++] = next;
                model.forward(new int[] {next});
                break;
            }
            if (tokenizer.skipGeneratedPiece(next)) {
                forwardedIds[forwarded++] = next;
                logits = model.forward(new int[] {next});
                continue;
            }
            String piece = tokenizer.tryDecodePiece(next);
            if (piece == null) {
                forwardedIds[forwarded++] = next;
                logits = model.forward(new int[] {next});
                continue;
            }
            completion.append(piece);
            if (onPiece != null) {
                onPiece.accept(piece);
            }
            generatedIds[generated] = next;
            forwardedIds[forwarded++] = next;
            generated++;
            samplerState.record(next);
            logits = model.forward(new int[] {next});
        }

        int[] trimmed = Arrays.copyOf(generatedIds, generated);
        int[] forwardedTrimmed = Arrays.copyOf(forwardedIds, forwarded);
        cachedTokens = concat(promptIds, forwardedTrimmed);
        return new ChatGenerationResult(
                completion.toString(),
                generated,
                options.mode().name(),
                startPos,
                trimmed,
                forwardedTrimmed);
    }

    /** Chunked prefill: forward {@code tokens[startPos:]} in blocks, return last-token logits. */
    static float[] prefillChunked(ChatModel model, int[] tokens, int startPos, int chunkSize) {
        if (startPos >= tokens.length) {
            throw new IllegalArgumentException("startPos beyond prompt length");
        }
        float[] logits = null;
        int pos = startPos;
        int chunk = Math.max(1, chunkSize);
        while (pos < tokens.length) {
            int end = Math.min(pos + chunk, tokens.length);
            logits = model.forward(Arrays.copyOfRange(tokens, pos, end));
            pos = end;
        }
        return logits;
    }

    private float[] prefillChunked(int[] tokens, int startPos, int chunkSize) {
        return prefillChunked(model, tokens, startPos, chunkSize);
    }

    private static int[] concat(int[] left, int[] right) {
        int[] out = new int[left.length + right.length];
        System.arraycopy(left, 0, out, 0, left.length);
        System.arraycopy(right, 0, out, left.length, right.length);
        return out;
    }
}
