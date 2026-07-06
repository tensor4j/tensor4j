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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufTensorSource;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Qwen 2.5 + {@link ChatHistoryMode#LLAMA} pipeline tests.
 *
 * <p>{@code history=llama} is llama.cpp <em>delta KV</em> bookkeeping — not Llama 3 chat headers.
 * The prompt format must still come from {@link ChatTemplate#QWEN2} / ChatML tokens.
 */
class QwenLlamaHistoryModeTest {

    private static final String LLAMA_HEADER = "<|" + "start_header_id" + "|>";
    private static final String QWEN_IM_START = "<|im_start|>";

    @Test
    void llamaHistoryModeNameIsDeltaKvNotLlamaTemplate() {
        assertEquals(InferCompatMode.LLAMA_CPP.defaultHistoryMode(), ChatHistoryMode.parseName("llama"));
        assertNotEquals(ChatTemplate.LLAMA3, ChatTemplate.QWEN2);
    }

    @Test
    void qwenApplierStringFormatUsesChatMlNotLlamaHeaders() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        String formatted = applier.apply(List.of(new ChatMessage("user", "hello")), true);

        assertTrue(formatted.contains(QWEN_IM_START), "Qwen must use ChatML im_start");
        assertFalse(formatted.contains(LLAMA_HEADER), "Qwen must not use Llama 3 header tokens");
    }

    @Test
    void qwenLlamaModeTokenIdsMatchQwen2Template() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> messages = List.of(new ChatMessage("user", "hello"));

        int[] llamaModeIds = applier.tokenIds(tokenizer, messages, true);
        int[] qwenTemplateIds = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "hello");
        assertArrayEquals(qwenTemplateIds, llamaModeIds);
    }

    @Test
    void qwenSessionTokensContainImStartNotLlamaHeaders() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);
        generator.continueConversation("hello", ChatTemplate.QWEN2);

        ChatTokenizer tokenizer = model.tokenizer();
        int imStart = tokenizer.tokenIdForText(QWEN_IM_START);
        int[] session = generator.sessionTokenIds();
        assertTrue(contains(session, imStart), "closed session must contain ChatML im_start");
        assertEquals(tokenizer.tokenIdForText("system"), session[1], "session must open with system role token");

        String decoded = tokenizer.decode(session);
        assertTrue(decoded.startsWith(QWEN_IM_START + "system\n"), () -> "session decode must lead with system turn: " + decoded);
        assertFalse(decoded.contains(LLAMA_HEADER), "session must not contain Llama header text");
    }

    @Test
    void qwenTokenizerInfersQwen2TemplateWhenEnvUnset() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        assertEquals(BpePreType.QWEN2, tokenizer.preType());
        assertEquals(ChatTemplate.QWEN2, ChatTemplate.fromTokenizer(tokenizer));
    }

    @Test
    void envTemplateOverrideCanForceLlama3OnQwenTokenizer() {
        String env = System.getenv("TENSOR4J_CHAT_TEMPLATE");
        if (env == null || env.isBlank()) {
            return;
        }
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        ChatTemplate selected = ChatTemplate.fromEnvironmentOrTokenizer(tokenizer);
        ChatTemplate inferred = ChatTemplate.fromTokenizer(tokenizer);
        if (!env.trim().equalsIgnoreCase("qwen2") && !env.trim().equalsIgnoreCase("qwen")) {
            assertNotEquals(
                    inferred,
                    selected,
                    "TENSOR4J_CHAT_TEMPLATE env override must not silently match tokenizer inference");
        }
    }

    @Test
    void forcingLlama3TemplateOnQwenTokenizerBreaksPromptParity() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        int[] correct = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "hello");
        int[] misconfigured = ChatTemplate.LLAMA3.encodePromptForGeneration(tokenizer, "hello");
        assertNotEquals(
                correct.length,
                misconfigured.length,
                "TENSOR4J_CHAT_TEMPLATE=llama3 on a Qwen GGUF must not match qwen2 prompts");
        assertFalse(Arrays.equals(correct, misconfigured));
    }

    @Test
    void qwenIncrementalDeltaPrefillMatchesColdFullTemplate() {
        GgufTensorSource weights = MiniChatGgufBuilder.buildQwen2TemplateModel();
        ChatModel warm = ChatModel.fromGguf(weights);
        ChatTokenizer tokenizer = warm.tokenizer();
        ChatGenerator generator = new ChatGenerator(
                warm, ChatGenerationOptions.greedy(tokenizer, 2), ChatHistoryMode.LLAMA);

        generator.continueConversation("Hi", ChatTemplate.QWEN2);
        generator.continueConversation("Yo", ChatTemplate.QWEN2);

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> planned = new ArrayList<>(generator.messages());
        planned.add(new ChatMessage("user", "Sup"));
        int prev = generator.templatePrevTokens();
        int[] full = applier.tokenIds(tokenizer, planned, true);
        int[] delta = applier.tokenDeltaSince(tokenizer, planned, true, prev);

        assertTrue(delta.length > 0);
        float[] warmLogits = warm.forward(delta);

        ChatModel cold = ChatModel.fromGguf(weights);
        cold.resetCache();
        float[] coldLogits = cold.forward(full);

        if (full.length <= warm.config().nCtx()) {
            assertArrayEquals(
                    coldLogits,
                    warmLogits,
                    1e-4f,
                    "Qwen delta prefill on warm KV must match stateless full-template forward");
        }
    }

    @Test
    void qwenLegacyFullReplayMatchesColdPrefillOnExtendedPrompt() {
        GgufTensorSource weights = MiniChatGgufBuilder.buildQwen2TemplateModel();
        ChatModel warm = ChatModel.fromGguf(weights);
        ChatGenerationOptions options = ChatGenerationOptions.greedy(warm.tokenizer(), 2);
        ChatGenerator generator = new ChatGenerator(warm, options, ChatHistoryMode.LEGACY);

        generator.continueConversation("Hi", ChatTemplate.QWEN2);
        int[] extended = generator.planPromptIds("Yo", ChatTemplate.QWEN2);
        int startPos = ChatGenerator.sharedPrefixLength(extended, generator.cachedTokenIds());
        assertTrue(startPos > 0);

        float[] warmLogits = ChatGenerator.prefillChunked(warm, extended, startPos, 2);

        ChatModel cold = ChatModel.fromGguf(weights);
        cold.resetCache();
        float[] coldLogits = ChatGenerator.prefillChunked(cold, extended, 0, options.prefillChunkSize());

        if (extended.length <= warm.config().nCtx()) {
            assertArrayEquals(coldLogits, warmLogits, 1e-4f, "stateless full replay must match warm KV resume");
        }
    }

    @Test
    void qwenReencodingAssistantCanDriftFromSampledIds() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);
        ChatGenerationResult result = generator.continueConversation("Hi", ChatTemplate.QWEN2);

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(model.tokenizer());
        List<ChatMessage> withReencode = List.of(
                new ChatMessage("user", "Hi"),
                new ChatMessage("assistant", result.text()));
        List<ChatMessage> withIds = List.of(
                new ChatMessage("user", "Hi"),
                new ChatMessage("assistant", result.text(), result.forwardedTokenIds()));

        int[] reencoded = applier.tokenIds(model.tokenizer(), withReencode, false);
        int[] fromIds = applier.tokenIds(model.tokenizer(), withIds, false);
        assertArrayEquals(generator.kvTokenIds(), fromIds);
        if (!Arrays.equals(reencoded, fromIds)) {
            assertNotEquals(reencoded.length, fromIds.length);
        }
    }

    @Test
    void turnOneAssistantImEndPrecedesTurnTwoUserDelta() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatTokenizer tokenizer = model.tokenizer();
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(tokenizer, 4), ChatHistoryMode.LLAMA);
        generator.continueConversation("Hi", ChatTemplate.QWEN2);

        int prev = generator.templatePrevTokens();
        int[] closed = generator.kvTokenIds();
        int endTurn = tokenizer.endTurnId();
        int newline = tokenizer.encode("\n")[0];
        assertTrue(prev >= 2, "closed template must have a boundary before turn 2");
        assertEquals(endTurn, closed[closed.length - 2], "session must end with assistant im_end");
        assertEquals(newline, closed[closed.length - 1], "session must end with newline after im_end");
        assertEquals(prev, closed.length);
        if (closed.length <= model.config().nCtx()) {
            assertEquals(prev, generator.kvLength(), "KV must match session when within n_ctx");
        }

        int[] modelInput = generator.modelInputTokenIds("Yo", ChatTemplate.QWEN2);
        assertEquals(endTurn, modelInput[prev - 2], "model input must have turn-1 assistant im_end before turn-2 user");
        assertEquals(newline, modelInput[prev - 1]);
        int[] userHeader = ChatTemplate.QWEN2.encodeRole(tokenizer, "user");
        assertArrayEquals(userHeader, Arrays.copyOfRange(modelInput, prev, prev + userHeader.length));

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> planned = new ArrayList<>(generator.messages());
        planned.add(new ChatMessage("user", "Yo"));
        int[] full = applier.tokenIds(tokenizer, planned, true);
        if (ChatGenerator.kvCacheEnabled()) {
            assertArrayEquals(Arrays.copyOfRange(full, prev, full.length), generator.planPromptIds("Yo", ChatTemplate.QWEN2));
        } else {
            assertArrayEquals(full, generator.planPromptIds("Yo", ChatTemplate.QWEN2));
            assertArrayEquals(full, modelInput);
        }
    }

    @Test
    void qwenMultiTurnDeltaIsSuffixOfFullTemplate() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LLAMA);
        generator.continueConversation("one", ChatTemplate.QWEN2);

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(model.tokenizer());
        List<ChatMessage> planned = new ArrayList<>(generator.messages());
        planned.add(new ChatMessage("user", "two"));
        int prev = generator.templatePrevTokens();
        int[] full = applier.tokenIds(model.tokenizer(), planned, true);
        int[] delta = applier.tokenDeltaSince(model.tokenizer(), planned, true, prev);
        assertArrayEquals(Arrays.copyOfRange(full, prev, full.length), delta);
    }

    @Test
    void defaultQualitySamplingWithinSanityBounds() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        ChatGenerationOptions options = ChatGenerationOptions.quality(tokenizer);
        assertEquals(0.7f, options.temperature(), 1e-6f);
        assertEquals(0.9f, options.topP(), 1e-6f);
        assertEquals(40, options.topK());
        assertTrue(options.minNewTokens() >= 2);
    }

    private static boolean contains(int[] array, int value) {
        for (int id : array) {
            if (id == value) {
                return true;
            }
        }
        return false;
    }
}
