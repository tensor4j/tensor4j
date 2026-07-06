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

import java.util.List;
import java.util.Locale;

/**
 * Renders chat prompts the way llama.cpp {@code llama_chat_apply_template} does for common
 * instruct models — built-in templates for Llama 3 and Qwen2 match HuggingFace / GGUF defaults.
 */
public final class LlamaCppChatApplier {

    private static final String LLAMA3_BEGIN = "<|begin_of_text|>";
    private static final String LLAMA3_HEADER_START = "<|" + "start_header_id" + "|>";
    private static final String LLAMA3_HEADER_END = "<|" + "end_header_id" + "|>";
    private static final String LLAMA3_EOT = "<|eot_id|>";

    private static final String QWEN_IM_START = "<|im_start|>";
    private static final String QWEN_IM_END = "<|" + "im_end" + "|>";

    private final BpePreType preType;
    private final String ggufTemplate;

    private LlamaCppChatApplier(BpePreType preType, String ggufTemplate) {
        this.preType = preType;
        this.ggufTemplate = ggufTemplate;
    }

    public static LlamaCppChatApplier fromTokenizer(ChatTokenizer tokenizer) {
        return new LlamaCppChatApplier(tokenizer.preType(), tokenizer.chatTemplate());
    }

    /**
     * Full formatted prompt string ({@code llama_chat_apply_template}).
     *
     * @param addGenerationPrompt when true, opens the assistant generation slot
     */
    public String apply(List<ChatMessage> messages, boolean addGenerationPrompt) {
        if (preType == BpePreType.QWEN2 || preType == BpePreType.QWEN35) {
            return applyQwen2(messages, addGenerationPrompt);
        }
        if (preType == BpePreType.LLAMA3) {
            return applyLlama3(messages, addGenerationPrompt);
        }
        if (ggufTemplate != null && !ggufTemplate.isBlank()) {
            throw new UnsupportedOperationException(
                    "custom tokenizer.chat_template requires llama3 or qwen2 pre type (got "
                            + preType
                            + "); set TENSOR4J_CHAT_HISTORY_MODE=legacy");
        }
        throw new UnsupportedOperationException(
                "llama history mode requires llama3 or qwen2 tokenizer (pre="
                        + preType
                        + "); set TENSOR4J_CHAT_HISTORY_MODE=legacy");
    }

    /** Delta prompt for the next decode ({@code simple-chat} {@code formatted[prev_len:new_len]}). */
    public String deltaSince(List<ChatMessage> messages, boolean addGenerationPrompt, int prevLen) {
        String full = apply(messages, addGenerationPrompt);
        if (prevLen < 0 || prevLen > full.length()) {
            throw new IllegalArgumentException("prevLen out of range for formatted prompt");
        }
        return full.substring(prevLen);
    }

    /** Length after assistant reply is committed ({@code add_generation_prompt=false}). */
    public int lengthAfterAssistantTurn(List<ChatMessage> messages) {
        return apply(messages, false).length();
    }

    /**
     * Token ids for the full formatted chat ({@code llama_chat_apply_template} + tokenize).
     *
     * <p>Uses structured special-token encoding (same as legacy {@link ChatTemplate}) — plain
     * {@link ChatTokenizer#encode(String)} on template strings BPE-splits {@code <|eot_id|>} etc.
     */
    public int[] tokenIds(ChatTokenizer tokenizer, List<ChatMessage> messages, boolean addGenerationPrompt) {
        ChatTemplate template = ChatTemplate.fromTokenizer(tokenizer);
        int[] ids = template.encodePrefix(tokenizer);
        boolean hasLeadingSystem = !messages.isEmpty() && "system".equals(messages.get(0).role());
        ids = concat(ids, template.encodeDefaultSystemTurnIfMissing(tokenizer, hasLeadingSystem));
        for (ChatMessage message : messages) {
            int[] role = template.encodeRole(tokenizer, message.role());
            int[] body = messageBodyTokenIds(tokenizer, message);
            int[] turn = concat(role, body);
            ids = concat(ids, turn, template.encodeEndTurnAfter(tokenizer, turn));
        }
        if (addGenerationPrompt) {
            ids = concat(ids, template.encodeAssistantPrime(tokenizer));
        }
        return ids;
    }

    /** Token delta since the last committed turn ({@code tokenIds[prevTokenCount:]}). */
    public int[] tokenDeltaSince(
            ChatTokenizer tokenizer, List<ChatMessage> messages, boolean addGenerationPrompt, int prevTokenCount) {
        int[] full = tokenIds(tokenizer, messages, addGenerationPrompt);
        if (prevTokenCount < 0 || prevTokenCount > full.length) {
            throw new IllegalArgumentException("prevTokenCount out of range for chat token ids");
        }
        return java.util.Arrays.copyOfRange(full, prevTokenCount, full.length);
    }

    /** Token count after assistant reply is committed ({@code add_generation_prompt=false}). */
    public int tokenCountAfterAssistantTurn(ChatTokenizer tokenizer, List<ChatMessage> messages) {
        return tokenIds(tokenizer, messages, false).length;
    }

    /** Prefer sampled ids for assistant turns — re-encoding decoded text can drift from KV. */
    private static int[] messageBodyTokenIds(ChatTokenizer tokenizer, ChatMessage message) {
        if ("assistant".equals(message.role())
                && message.generatedTokenIds() != null
                && InferCompatMode.fromEnvironment().useSampledAssistantTokenIds()) {
            return message.generatedTokenIds();
        }
        return tokenizer.encode(message.content().trim());
    }

    private static int[] concat(int[]... parts) {
        int len = 0;
        for (int[] part : parts) {
            len += part.length;
        }
        int[] out = new int[len];
        int pos = 0;
        for (int[] part : parts) {
            System.arraycopy(part, 0, out, pos, part.length);
            pos += part.length;
        }
        return out;
    }

    private static String applyLlama3(List<ChatMessage> messages, boolean addGenerationPrompt) {
        StringBuilder out = new StringBuilder();
        if (messages.isEmpty() || !"system".equals(messages.get(0).role())) {
            out.append(LLAMA3_BEGIN);
        }
        for (ChatMessage message : messages) {
            String role = message.role().toLowerCase(Locale.ROOT);
            if (!"user".equals(role) && !"assistant".equals(role) && !"system".equals(role)) {
                throw new IllegalArgumentException("unsupported role: " + message.role());
            }
            out.append(LLAMA3_HEADER_START)
                    .append(role)
                    .append(LLAMA3_HEADER_END)
                    .append("\n\n")
                    .append(message.content().trim())
                    .append(LLAMA3_EOT);
        }
        if (addGenerationPrompt) {
            out.append(LLAMA3_HEADER_START)
                    .append("assistant")
                    .append(LLAMA3_HEADER_END)
                    .append("\n\n");
        }
        return out.toString();
    }

    private static String applyQwen2(List<ChatMessage> messages, boolean addGenerationPrompt) {
        StringBuilder out = new StringBuilder();
        for (ChatMessage message : messages) {
            out.append(QWEN_IM_START)
                    .append(message.role())
                    .append('\n')
                    .append(message.content().trim())
                    .append(QWEN_IM_END)
                    .append('\n');
        }
        if (addGenerationPrompt) {
            out.append(QWEN_IM_START).append("assistant").append('\n');
        }
        return out.toString();
    }
}
