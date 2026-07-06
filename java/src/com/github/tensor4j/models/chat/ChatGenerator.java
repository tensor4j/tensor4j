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
    /** Every token actually forwarded into KV (debug + turn-close truth; not template-rebuilt). */
    private int[] kvTokenIds = new int[0];

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
        kvTokenIds = new int[0];
        messages.clear();
        templatePrevTokens = 0;
        samplerState.reset();
        model.resetCache();
    }

    private void ensureDefaultSystemMessage() {
        if (!ChatTemplate.defaultSystemTurnEnabled()) {
            return;
        }
        if (messages.isEmpty() && ChatTemplate.fromTokenizer(tokenizer) == ChatTemplate.QWEN2) {
            messages.add(new ChatMessage("system", ChatTemplate.defaultSystemPromptText()));
        }
    }

    public ChatGenerationResult generate(String prompt, ChatTemplate template) {
        if (historyMode == ChatHistoryMode.LLAMA && chatApplier != null) {
            messages.clear();
            templatePrevTokens = 0;
            kvTokenIds = new int[0];
            model.resetCache();
            ensureDefaultSystemMessage();
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
        prepareForTurn(!kvCacheEnabled());
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
        prepareForTurn(!kvCacheEnabled());
        if (historyMode == ChatHistoryMode.LLAMA) {
            return continueConversationLlama(userMessage, onPiece);
        }
        int[] promptIds = buildPromptIdsLegacy(userMessage, template);
        ChatGenerationResult result = generate(promptIds, kvReuseEnabled(), false, onPiece);
        finishTurnLegacy(promptIds, result.forwardedTokenIds(), template);
        return result;
    }

    private ChatGenerationResult continueConversationLlama(String userMessage, Consumer<String> onPiece) {
        ensureDefaultSystemMessage();
        messages.add(new ChatMessage("user", userMessage));
        int[] promptIds;
        boolean deltaAppend;
        if (kvCacheEnabled()) {
            if (templatePrevTokens > 0) {
                requireClosedAssistantBoundary(conversationTokens);
            }
            promptIds = templatePrevTokens == 0
                    ? chatApplier.tokenIds(tokenizer, messages, true)
                    : buildLlamaTurnPrefill(userMessage);
            deltaAppend = templatePrevTokens > 0;
        } else {
            prepareForTurn(true);
            promptIds = chatApplier.tokenIds(tokenizer, messages, true);
            deltaAppend = false;
        }
        if (ChatInferBufferPolicy.logPromptTextBeforeTokenize() && ChatTokenDebugLog.enabled()) {
            ChatTokenDebugLog.logPromptText(System.err, "prefill_prompt_text", chatApplier.apply(messages, true));
        }
        ChatGenerationResult result = generate(promptIds, false, deltaAppend, onPiece);
        finishTurnLlama(result.text(), result.forwardedTokenIds());
        return result;
    }

    /**
     * Cross-turn KV cache (llama delta prefill + legacy prefix reuse).
     *
     * <p>Default {@code false} ({@code TENSOR4J_CHAT_KV_CACHE} unset). Set {@code true} for llama.cpp-style
     * delta KV between turns.
     */
    public static boolean kvCacheEnabled() {
        return parseKvCacheEnabled(System.getenv("TENSOR4J_CHAT_KV_CACHE"));
    }

    static boolean parseKvCacheEnabled(String raw) {
        if (raw == null || raw.isBlank()) {
            return false;
        }
        return Boolean.parseBoolean(raw.trim());
    }

    /** Legacy prefix reuse within a single turn's full replay (requires {@link #kvCacheEnabled()}). */
    public static boolean kvReuseEnabled() {
        return kvCacheEnabled()
                && !Boolean.parseBoolean(System.getenv().getOrDefault("TENSOR4J_CHAT_NO_KV_REUSE", "false"));
    }

    /**
     * Clear model KV at each user turn (default {@code true} when {@link #kvCacheEnabled()} is off).
     *
     * <p>Env: {@code TENSOR4J_CHAT_RESET_MODEL_EACH_TURN}.
     */
    public static boolean resetModelEachTurn() {
        String raw = System.getenv("TENSOR4J_CHAT_RESET_MODEL_EACH_TURN");
        if (raw != null && !raw.isBlank()) {
            return Boolean.parseBoolean(raw.trim());
        }
        return !kvCacheEnabled();
    }

    /** Reset repetition-penalty / alpha sampler state each turn (default {@code true}). */
    public static boolean resetSamplerEachTurn() {
        return parseBoolEnv("TENSOR4J_CHAT_RESET_SAMPLER_EACH_TURN", true);
    }

    /**
     * Clone logits before sampling — guards against reused forward buffers (default {@code true}).
     *
     * <p>Env: {@code TENSOR4J_CHAT_CLONE_LOGITS} or {@code TENSOR4J_CHAT_CLONE_FORWARD_LOGITS}.
     *
     * @see ChatInferBufferPolicy#cloneForwardLogits()
     */
    public static boolean cloneLogitsBeforeSample() {
        return ChatInferBufferPolicy.cloneForwardLogits();
    }

    /**
     * Stop on {@code im_end} when {@code min_new_tokens} would otherwise spin until {@code max_new_tokens}
     * (default {@code true}).
     *
     * <p>Env: {@code TENSOR4J_CHAT_BREAK_EOG_SPIN}.
     */
    public static boolean breakEogSpinLoop() {
        return parseBoolEnv("TENSOR4J_CHAT_BREAK_EOG_SPIN", true);
    }

    /** Throw when {@code kvTokenIds} != closed template after turn close (default {@code false}). */
    public static boolean strictSessionDrift() {
        return parseBoolEnv("TENSOR4J_CHAT_STRICT_SESSION", false);
    }

    static boolean parseBoolEnv(String key, boolean defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        return Boolean.parseBoolean(raw.trim());
    }

    private void prepareForTurn(boolean coldReplay) {
        if (resetSamplerEachTurn()) {
            samplerState.reset();
        }
        if (coldReplay || resetModelEachTurn()) {
            model.resetCache();
            kvTokenIds = new int[0];
            if (coldReplay) {
                templatePrevTokens = 0;
            }
        }
    }

    public int[] planPromptIds(String userMessage, ChatTemplate template) {
        if (historyMode == ChatHistoryMode.LLAMA) {
            List<ChatMessage> planned = new ArrayList<>(messages);
            if (planned.isEmpty()
                    && ChatTemplate.defaultSystemTurnEnabled()
                    && ChatTemplate.fromTokenizer(tokenizer) == ChatTemplate.QWEN2) {
                planned.add(new ChatMessage("system", ChatTemplate.defaultSystemPromptText()));
            }
            planned.add(new ChatMessage("user", userMessage));
            if (!kvCacheEnabled() || templatePrevTokens == 0) {
                return chatApplier.tokenIds(tokenizer, planned, true);
            }
            requireClosedAssistantBoundary(conversationTokens);
            return buildLlamaTurnPrefill(userMessage);
        }
        return buildPromptIdsLegacy(userMessage, template);
    }

    /**
     * Contiguous token sequence the model sees after prefill: closed session (with assistant {@code im_end})
     * immediately followed by this turn's user + assistant-prime tokens.
     *
     * <p>When {@link #kvCacheEnabled()} is off, equals {@link #planPromptIds} (full cold replay each turn).
     */
    public int[] modelInputTokenIds(String userMessage, ChatTemplate template) {
        int[] turnPrefill = planPromptIds(userMessage, template);
        if (historyMode == ChatHistoryMode.LLAMA && kvCacheEnabled() && templatePrevTokens > 0) {
            return concat(conversationTokens, turnPrefill);
        }
        return turnPrefill;
    }

    private int[] buildLlamaTurnPrefill(String userMessage) {
        ChatTemplate chatTemplate = ChatTemplate.fromTokenizer(tokenizer);
        return concat(
                chatTemplate.encodeUserTurn(tokenizer, userMessage),
                chatTemplate.encodeAssistantPrime(tokenizer));
    }

    private void requireClosedAssistantBoundary(int[] tokens) {
        ChatTemplate chatTemplate = ChatTemplate.fromTokenizer(tokenizer);
        if (!chatTemplate.usesStructuredTurns()) {
            return;
        }
        if (tokens.length < 2) {
            throw new IllegalStateException("session too short before next user turn");
        }
        int endId = tokenizer.endTurnId();
        int newlineId = trailingNewlineId(tokenizer);
        if (tokens[tokens.length - 2] != endId || tokens[tokens.length - 1] != newlineId) {
            throw new IllegalStateException(
                    "session must end with assistant im_end (id="
                            + endId
                            + ") then newline (id="
                            + newlineId
                            + ") before next user turn; got "
                            + tokens[tokens.length - 2]
                            + ", "
                            + tokens[tokens.length - 1]);
        }
    }

    private static int trailingNewlineId(ChatTokenizer tokenizer) {
        int[] encoded = tokenizer.encode("\n");
        return encoded.length > 0 ? encoded[0] : tokenizer.tokenIdForText("\n");
    }

    public int[] sessionTokenIds() {
        return conversationTokens.clone();
    }

    /** Token ids actually forwarded into KV (includes assistant {@code im_end} when turn closed). */
    public int[] kvTokenIds() {
        return kvTokenIds.clone();
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

    /** Package-visible for buffer-isolation tests. */
    int[] cachedTokensInternal() {
        return cachedTokens;
    }

    /** Package-visible for buffer-isolation tests. */
    int[] conversationTokensInternal() {
        return conversationTokens;
    }

    /** Package-visible for buffer-isolation tests. */
    float[] forwardTrackedForTests(int[] tokens) {
        return forwardTracked(tokens);
    }

    private int[] buildPromptIdsLegacy(String userMessage, ChatTemplate template) {
        int[] turn = concat(
                template.encodeUserTurn(tokenizer, userMessage),
                template.encodeAssistantPrime(tokenizer));
        return conversationTokens.length == 0
                ? concat(
                        template.encodePrefix(tokenizer),
                        concat(template.encodeDefaultSystemTurnIfMissing(tokenizer, false), turn))
                : concat(conversationTokens, turn);
    }

    private void finishTurnLlama(String assistantText, int[] forwarded) {
        if (InferCompatMode.fromEnvironment().useSampledAssistantTokenIds()) {
            messages.add(new ChatMessage("assistant", assistantText, forwarded));
        } else {
            messages.add(new ChatMessage("assistant", assistantText));
        }
        ChatTemplate template = ChatTemplate.fromTokenizer(tokenizer);
        int[] suffix = template.encodeEndTurnAfter(tokenizer, forwarded);
        if (suffix.length > 0) {
            forwardTracked(suffix);
        }
        conversationTokens = kvTokenIds.clone();
        cachedTokens = conversationTokens.clone();
        templatePrevTokens = kvTokenIds.length;
        int kv = model.kvLength();
        int[] expectedClosed = chatApplier.tokenIds(tokenizer, messages, false);
        if (ChatTokenDebugLog.enabled()) {
            ChatTokenDebugLog.log(System.err, "assistant_forwarded", tokenizer, forwarded);
            ChatTokenDebugLog.log(System.err, "turn_close_suffix", tokenizer, suffix);
            ChatTokenDebugLog.log(System.err, "kv_after_turn_close", tokenizer, kvTokenIds);
            ChatTokenDebugLog.logKvHandoff(System.err, kv, templatePrevTokens, suffix);
        }
        if (kv != templatePrevTokens && !(templatePrevTokens > model.config().nCtx() && kv == model.config().nCtx())) {
            throw new IllegalStateException(
                    "ChatML KV length drift after turn close: kv="
                            + kv
                            + " forwarded="
                            + templatePrevTokens
                            + " nCtx="
                            + model.config().nCtx());
        }
        if (strictSessionDrift()
                && templatePrevTokens <= model.config().nCtx()
                && !Arrays.equals(expectedClosed, kvTokenIds)) {
            throw new IllegalStateException(
                    "ChatML KV token drift after turn close: forwarded ids != closed template");
        }
    }

    private static int lastToken(int[] ids) {
        return ids.length == 0 ? -1 : ids[ids.length - 1];
    }

    private void finishTurnLegacy(int[] promptIds, int[] forwarded, ChatTemplate template) {
        int[] closed = closeAssistantTurn(forwarded, template);
        conversationTokens = concat(promptIds, closed);
        cachedTokens = ChatInferBufferPolicy.isolateSessionTokens(conversationTokens);
    }

    private int[] closeAssistantTurn(int[] forwarded, ChatTemplate template) {
        if (!template.usesStructuredTurns()) {
            return forwarded;
        }
        int[] suffix = template.encodeEndTurnAfter(tokenizer, forwarded);
        if (suffix.length > 0) {
            forwardTracked(suffix);
            return concat(forwarded, suffix);
        }
        return forwarded;
    }

    private float[] forwardTracked(int[] tokens) {
        if (tokens.length == 0) {
            throw new IllegalArgumentException("empty forward");
        }
        int[] forwardTokens = ChatInferBufferPolicy.isolatePromptTokens(tokens);
        float[] logits = ChatInferBufferPolicy.isolateForwardLogits(model.forward(forwardTokens));
        ChatInferBufferPolicy.logIdentity(System.err, "forward_logits", logits);
        ChatInferBufferPolicy.logIdentity(System.err, "forward_tokens", forwardTokens);
        kvTokenIds = concat(kvTokenIds, forwardTokens);
        return logits;
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

        promptIds = ChatInferBufferPolicy.isolatePromptTokens(promptIds);
        ChatInferBufferPolicy.logIdentity(System.err, "generate_prompt_ids", promptIds);

        boolean debug = ChatTokenDebugLog.enabled();
        java.util.ArrayList<ChatGenerationStep> stepLog = debug ? new java.util.ArrayList<>() : null;

        int startPos = 0;
        if (reusePrefix && cachedTokens.length > 0) {
            startPos = sharedPrefixLength(promptIds, cachedTokens);
        }
        if (!deltaAppend && (startPos == 0 || model.kvLength() != startPos)) {
            model.resetCache();
            kvTokenIds = new int[0];
            startPos = 0;
        }

        float[] logits = prefillChunkedTracked(promptIds, startPos, options.prefillChunkSize());

        StringBuilder completion = new StringBuilder();
        int[] generatedIds = new int[options.maxNewTokens()];
        int[] forwardedIds = new int[options.maxNewTokens()];
        int generated = 0;
        int forwarded = 0;
        boolean llamaStop = historyMode == ChatHistoryMode.LLAMA;
        boolean structuredChat = ChatTemplate.fromTokenizer(tokenizer).usesStructuredTurns();
        ChatGenerationStopReason stopReason = ChatGenerationStopReason.MAX_TOKENS;
        int stopTokenId = -1;
        int stepsTaken = 0;
        for (int step = 0; step < options.maxNewTokens(); step++) {
            stepsTaken = step + 1;
            float[] sampleLogits = logits;
            int next = structuredChat
                    ? ChatSampler.sampleExcludingStructureTokens(
                            sampleLogits, options, generated, samplerState, samplingRng, tokenizer)
                    : ChatSampler.sample(sampleLogits, options, generated, samplerState, samplingRng);
            if (tokenizer.isEndOfGeneration(next) && generated < options.minNewTokens()) {
                next = ChatSampler.sampleExcludingEndTokens(
                        sampleLogits, options, generated, samplerState, samplingRng);
            }
            boolean endOfGen = tokenizer.isEndOfGeneration(next);
            boolean stop = llamaStop
                    ? endOfGen && generated >= options.minNewTokens()
                    : TinygradGenerateReference.shouldStop(next, tokenizer.eosId(), generated, options.minNewTokens());
            if (stop) {
                stopReason = ChatGenerationStopReason.forEndToken(tokenizer, next);
                stopTokenId = next;
                forwardedIds[forwarded++] = next;
                forwardTracked(new int[] {next});
                logStep(stepLog, step, next, null, false, true);
                break;
            }
            if (endOfGen) {
                if (breakEogSpinLoop() && llamaStop) {
                    stopReason = ChatGenerationStopReason.forEndToken(tokenizer, next);
                    stopTokenId = next;
                    forwardedIds[forwarded++] = next;
                    forwardTracked(new int[] {next});
                    logStep(stepLog, step, next, null, false, true);
                    break;
                }
                logStep(stepLog, step, next, null, false, true);
                continue;
            }
            if (tokenizer.skipGeneratedPiece(next)) {
                logStep(stepLog, step, next, tokenizer.tokenText(next), false, endOfGen);
                if (!structuredChat || !tokenizer.isChatStructureInjectionToken(next)) {
                    forwardedIds[forwarded++] = next;
                    logits = forwardTracked(new int[] {next});
                }
                continue;
            }
            String piece = tokenizer.tryDecodePiece(next);
            if (piece == null) {
                logStep(stepLog, step, next, null, false, endOfGen);
                if (!structuredChat || !tokenizer.isChatStructureInjectionToken(next)) {
                    forwardedIds[forwarded++] = next;
                    logits = forwardTracked(new int[] {next});
                }
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
            logits = forwardTracked(new int[] {next});
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

    private float[] prefillChunkedTracked(int[] tokens, int startPos, int chunkSize) {
        if (startPos >= tokens.length) {
            throw new IllegalArgumentException("startPos beyond prompt length");
        }
        float[] logits = null;
        int pos = startPos;
        int chunk = Math.max(1, chunkSize);
        while (pos < tokens.length) {
            int end = Math.min(pos + chunk, tokens.length);
            logits = forwardTracked(Arrays.copyOfRange(tokens, pos, end));
            pos = end;
        }
        return logits;
    }

    private static int[] concat(int[] left, int[] right) {
        int[] out = new int[left.length + right.length];
        System.arraycopy(left, 0, out, 0, left.length);
        System.arraycopy(right, 0, out, left.length, right.length);
        return out;
    }
}
