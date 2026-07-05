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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

/**
 * Stateful decode loop — chunked prefill, KV cache, streaming.
 *
 * <p>{@link ChatHistoryMode#LLAMA}: llama.cpp {@code simple-chat} delta tokenization (default).
 * <p>{@link ChatHistoryMode#LEGACY}: tinygrad {@code apps/llm.py} full token replay + prefix reuse.
 */
public final class ChatGenerator {

    private final ChatModel model;
    private final ChatGenerationOptions options;
    private final ChatTokenizer tokenizer;
    private final ChatSamplingRng samplingRng;
    private final ChatSamplerState samplerState;
    private final ChatHistoryMode historyMode;
    private final LlamaCppChatApplier chatApplier;

    private int[] cachedTokens = new int[0];
    private int[] conversationTokens = new int[0];

    private final List<ChatMessage> messages = new ArrayList<>();
    private int templatePrevTokens;

    public ChatGenerator(ChatModel model, ChatGenerationOptions options) {
        this(model, options, ChatHistoryMode.fromEnvironment());
    }

    public ChatGenerator(ChatModel model, ChatGenerationOptions options, ChatHistoryMode historyMode) {
        this.model = model;
        this.options = options;
        this.tokenizer = model.tokenizer();
        this.samplingRng = ChatSamplingRng.fromOptions(options);
        this.samplerState = new ChatSamplerState(tokenizer.vocabSize());
        this.historyMode = historyMode;
        this.chatApplier = historyMode == ChatHistoryMode.LLAMA ? LlamaCppChatApplier.fromTokenizer(tokenizer) : null;
    }

    public ChatHistoryMode historyMode() {
        return historyMode;
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

    static boolean[] forwardPrefillFlags(int promptLen, int chunkSize, int yieldCount) {
        int[] slices = prefillSliceLengths(promptLen, 0, chunkSize);
        boolean[] flags = new boolean[slices.length + Math.max(0, yieldCount - 1)];
        Arrays.fill(flags, 0, slices.length, true);
        return flags;
    }

    ChatGenerationResult generateWithKvReuse(int[] promptIds) {
        return generate(promptIds, true, false, null);
    }

    public void resetSession() {
        cachedTokens = new int[0];
        conversationTokens = new int[0];
        messages.clear();
        templatePrevTokens = 0;
        samplerState.reset();
        model.resetCache();
    }

    public ChatGenerationResult generate(String prompt, ChatTemplate template) {
        if (historyMode == ChatHistoryMode.LLAMA && chatApplier != null) {
            messages.clear();
            templatePrevTokens = 0;
            model.resetCache();
            messages.add(new ChatMessage("user", prompt));
            int[] promptIds = chatApplier.tokenIds(tokenizer, messages, true);
            ChatGenerationResult result = generate(promptIds, false, false, null);
            finishTurnLlama(result.text(), result.forwardedTokenIds());
            return result;
        }
        return generate(template.encodePromptForGeneration(tokenizer, prompt), false, false, null);
    }

    public ChatGenerationResult generate(int[] promptIds) {
        return generate(promptIds, false, false, null);
    }

    public ChatGenerationResult continueConversation(String userMessage, ChatTemplate template) {
        samplerState.reset();
        if (historyMode == ChatHistoryMode.LLAMA) {
            return continueConversationLlama(userMessage, null);
        }
        int[] promptIds = buildPromptIdsLegacy(userMessage, template);
        ChatGenerationResult result = generate(promptIds, kvReuseEnabled(), false, null);
        finishTurnLegacy(promptIds, result.forwardedTokenIds(), template);
        return result;
    }

    public ChatGenerationResult continueConversationStreaming(
            String userMessage, ChatTemplate template, Consumer<String> onPiece) {
        samplerState.reset();
        if (historyMode == ChatHistoryMode.LLAMA) {
            return continueConversationLlama(userMessage, onPiece);
        }
        int[] promptIds = buildPromptIdsLegacy(userMessage, template);
        ChatGenerationResult result = generate(promptIds, kvReuseEnabled(), false, onPiece);
        finishTurnLegacy(promptIds, result.forwardedTokenIds(), template);
        return result;
    }

    private ChatGenerationResult continueConversationLlama(String userMessage, Consumer<String> onPiece) {
        messages.add(new ChatMessage("user", userMessage));
        int[] promptIds = chatApplier.tokenDeltaSince(tokenizer, messages, true, templatePrevTokens);
        ChatGenerationResult result = generate(promptIds, false, true, onPiece);
        finishTurnLlama(result.text(), result.forwardedTokenIds());
        return result;
    }

    public static boolean kvReuseEnabled() {
        return !Boolean.parseBoolean(System.getenv().getOrDefault("TENSOR4J_CHAT_NO_KV_REUSE", "false"));
    }

    public int[] planPromptIds(String userMessage, ChatTemplate template) {
        if (historyMode == ChatHistoryMode.LLAMA) {
            List<ChatMessage> planned = new ArrayList<>(messages);
            planned.add(new ChatMessage("user", userMessage));
            return chatApplier.tokenDeltaSince(tokenizer, planned, true, templatePrevTokens);
        }
        return buildPromptIdsLegacy(userMessage, template);
    }

    public int[] sessionTokenIds() {
        return conversationTokens.clone();
    }

    public int[] cachedTokenIds() {
        return cachedTokens.clone();
    }

    public List<ChatMessage> messages() {
        return List.copyOf(messages);
    }

    public int templatePrevTokens() {
        return templatePrevTokens;
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

    int[] sessionTokenIdsForTests() {
        return sessionTokenIds();
    }

    private int[] buildPromptIdsLegacy(String userMessage, ChatTemplate template) {
        int[] turn = concat(
                template.encodeUserTurn(tokenizer, userMessage),
                template.encodeAssistantPrime(tokenizer));
        return conversationTokens.length == 0
                ? concat(template.encodePrefix(tokenizer), turn)
                : concat(conversationTokens, turn);
    }

    private void finishTurnLlama(String assistantText, int[] forwarded) {
        if (InferCompatMode.fromEnvironment().useSampledAssistantTokenIds()) {
            messages.add(new ChatMessage("assistant", assistantText, forwarded));
        } else {
            messages.add(new ChatMessage("assistant", assistantText));
        }
        templatePrevTokens = chatApplier.tokenCountAfterAssistantTurn(tokenizer, messages);
        int[] closed = chatApplier.tokenIds(tokenizer, messages, false);
        conversationTokens = closed;
        cachedTokens = closed.clone();
        if (!tokenizer.isEndOfGeneration(lastToken(forwarded))) {
            int[] eot = ChatTemplate.fromTokenizer(tokenizer).encodeEndTurn(tokenizer);
            model.forward(eot);
        }
        if (Boolean.parseBoolean(System.getenv().getOrDefault("TENSOR4J_CHAT_DEBUG", "false"))) {
            int kv = model.kvLength();
            System.err.printf(
                    "chat session (llama): %d messages, %d template tokens, kv %d tokens%n",
                    messages.size(), templatePrevTokens, kv);
            if (kv != templatePrevTokens) {
                System.err.printf(
                        "chat session WARN: kv length (%d) != closed template tokens (%d) — delta/KV may drift%n",
                        kv, templatePrevTokens);
            }
        }
    }

    private static int lastToken(int[] ids) {
        return ids.length == 0 ? -1 : ids[ids.length - 1];
    }

    private void finishTurnLegacy(int[] promptIds, int[] forwarded, ChatTemplate template) {
        int[] closed = closeAssistantTurn(forwarded, template);
        conversationTokens = concat(promptIds, closed);
        cachedTokens = conversationTokens;
        if (Boolean.parseBoolean(System.getenv().getOrDefault("TENSOR4J_CHAT_DEBUG", "false"))) {
            System.err.printf(
                    "chat session (legacy): %d tokens, ends with eot=%s%n",
                    conversationTokens.length,
                    conversationTokens.length > 0
                            && conversationTokens[conversationTokens.length - 1] == tokenizer.eosId());
        }
    }

    private int[] closeAssistantTurn(int[] forwarded, ChatTemplate template) {
        if (!template.usesStructuredTurns()) {
            return forwarded;
        }
        if (forwarded.length > 0 && tokenizer.isEndOfGeneration(forwarded[forwarded.length - 1])) {
            return forwarded;
        }
        int[] eot = template.encodeEndTurn(tokenizer);
        model.forward(eot);
        return concat(forwarded, eot);
    }

    private ChatGenerationResult generate(
            int[] promptIds, boolean reusePrefix, boolean deltaAppend, Consumer<String> onPiece) {
        if (promptIds.length == 0) {
            return new ChatGenerationResult(
                    "",
                    0,
                    options.mode().name(),
                    0,
                    new int[0],
                    new int[0],
                    ChatGenerationStopReason.EMPTY_PROMPT,
                    -1,
                    options.maxNewTokens(),
                    new ChatGenerationStep[0]);
        }

        boolean debug = Boolean.parseBoolean(System.getenv().getOrDefault("TENSOR4J_CHAT_DEBUG", "false"));
        java.util.ArrayList<ChatGenerationStep> stepLog = debug ? new java.util.ArrayList<>() : null;

        int startPos = 0;
        if (reusePrefix && cachedTokens.length > 0) {
            startPos = sharedPrefixLength(promptIds, cachedTokens);
        }
        if (!deltaAppend && (startPos == 0 || model.kvLength() != startPos)) {
            model.resetCache();
            startPos = 0;
        }

        float[] logits = prefillChunked(promptIds, startPos, options.prefillChunkSize());

        StringBuilder completion = new StringBuilder();
        int[] generatedIds = new int[options.maxNewTokens()];
        int[] forwardedIds = new int[options.maxNewTokens()];
        int generated = 0;
        int forwarded = 0;
        boolean llamaStop = historyMode == ChatHistoryMode.LLAMA;
        ChatGenerationStopReason stopReason = ChatGenerationStopReason.MAX_TOKENS;
        int stopTokenId = -1;
        int stepsTaken = 0;
        for (int step = 0; step < options.maxNewTokens(); step++) {
            stepsTaken = step + 1;
            int next = ChatSampler.sample(logits, options, generated, samplerState, samplingRng);
            boolean endOfGen = tokenizer.isEndOfGeneration(next);
            boolean stop = llamaStop
                    ? endOfGen && generated >= options.minNewTokens()
                    : TinygradGenerateReference.shouldStop(next, tokenizer.eosId(), generated, options.minNewTokens());
            if (stop) {
                stopReason = ChatGenerationStopReason.forEndToken(tokenizer, next);
                stopTokenId = next;
                if (!llamaStop) {
                    forwardedIds[forwarded++] = next;
                    model.forward(new int[] {next});
                }
                logStep(stepLog, step, next, null, false, true);
                break;
            }
            if (tokenizer.skipGeneratedPiece(next)) {
                logStep(stepLog, step, next, tokenizer.tokenText(next), false, endOfGen);
                forwardedIds[forwarded++] = next;
                logits = model.forward(new int[] {next});
                continue;
            }
            String piece = tokenizer.tryDecodePiece(next);
            if (piece == null) {
                logStep(stepLog, step, next, null, false, endOfGen);
                forwardedIds[forwarded++] = next;
                logits = model.forward(new int[] {next});
                continue;
            }
            logStep(stepLog, step, next, piece, true, endOfGen);
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
        if (!deltaAppend) {
            cachedTokens = concat(promptIds, forwardedTrimmed);
        }
        int prefixReuse = deltaAppend ? 0 : startPos;
        ChatGenerationStep[] steps =
                stepLog == null ? new ChatGenerationStep[0] : stepLog.toArray(new ChatGenerationStep[0]);
        int tokensRemaining = Math.max(0, options.maxNewTokens() - stepsTaken);
        if (debug) {
            System.err.printf(
                    "chat generate: stop=%s token=%d visible=%d forwarded=%d max=%d min=%d remaining=%d%n",
                    stopReason,
                    stopTokenId,
                    generated,
                    forwarded,
                    options.maxNewTokens(),
                    options.minNewTokens(),
                    tokensRemaining);
        }
        return new ChatGenerationResult(
                completion.toString(),
                generated,
                options.mode().name(),
                prefixReuse,
                trimmed,
                forwardedTrimmed,
                stopReason,
                stopTokenId,
                tokensRemaining,
                steps);
    }

    private static void logStep(
            java.util.ArrayList<ChatGenerationStep> stepLog,
            int step,
            int tokenId,
            String piece,
            boolean visible,
            boolean endOfGeneration) {
        if (stepLog == null) {
            return;
        }
        stepLog.add(new ChatGenerationStep(step, tokenId, piece, visible, endOfGeneration));
        System.err.printf(
                "  step %d: id=%d visible=%s eog=%s text=%s%n",
                step, tokenId, visible, endOfGeneration, piece == null ? "(none)" : piece);
    }

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
