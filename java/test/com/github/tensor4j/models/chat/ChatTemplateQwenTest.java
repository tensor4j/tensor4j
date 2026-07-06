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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.reference.TinygradTokenizerGoldenCases;
import com.github.tensor4j.models.chat.reference.TinygradTokenizerReference;
import com.github.tensor4j.runtime.gguf.GgufFile;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Qwen2 ChatML prompt structure — parity with tinygrad {@code SimpleTokenizer} (qwen2 branch). */
class ChatTemplateQwenTest {

    private static final String IM_START = "<|im_start|>";

    @Test
    void qwenChatMlBoundariesMatchTinygradReferenceOnFixture() {
        ChatTokenizer tokenizer = qwenTokenizer();
        for (var golden : TinygradTokenizerGoldenCases.qwen2Cases()) {
            assertArrayEquals(
                    golden.expectedIds(),
                    TinygradTokenizerGoldenCases.expectedFor(tokenizer, golden),
                    golden.name());
        }
    }

    @Test
    void chatMlRoleAndEndTurnUseExistingEncodersWithNewlineTokens() {
        ChatTokenizer tokenizer = qwenTokenizer();
        int imStartId = tokenizer.tokenIdForText(IM_START);
        int imEndId = tokenizer.endTurnId();
        int newlineId = newlineId(tokenizer);

        int[] userRole = ChatTemplate.QWEN2.encodeRole(tokenizer, "user");
        int[] systemRole = ChatTemplate.QWEN2.encodeRole(tokenizer, "system");
        assertArrayEquals(TinygradTokenizerReference.role(tokenizer, "user"), userRole);
        assertArrayEquals(TinygradTokenizerReference.role(tokenizer, "system"), systemRole);
        assertEquals(imStartId, userRole[0], "role opens with im_start");
        assertEquals(imStartId, systemRole[0], "system role opens with im_start");
        assertEquals(tokenizer.tokenIdForText("user"), userRole[1], "role name is one vocab token");
        assertEquals(tokenizer.tokenIdForText("system"), systemRole[1], "system name is one vocab token");
        assertEquals(newlineId, userRole[userRole.length - 1], "role closes with newline token");
        assertEquals(newlineId, systemRole[systemRole.length - 1], "system header closes with newline token");

        int[] endTurn = ChatTemplate.QWEN2.encodeEndTurn(tokenizer);
        assertArrayEquals(TinygradTokenizerReference.endTurn(tokenizer, imEndId), endTurn);
        assertEquals(imEndId, endTurn[0], "turn closes with im_end");
        assertEquals(newlineId, endTurn[1], "im_end is followed by newline token");
        assertTrue(tokenizer.eosId() != imEndId, "fixture must split eos vs im_end like real Qwen GGUF");
    }

    @Test
    void chatMlPrefillMatchesExistingEncodersIncludingNewlineBoundaries() {
        ChatTokenizer tokenizer = qwenTokenizer();
        int imStartId = tokenizer.tokenIdForText(IM_START);
        int imEndId = tokenizer.endTurnId();
        int newlineId = newlineId(tokenizer);

        int[] systemRole = ChatTemplate.QWEN2.encodeRole(tokenizer, "system");
        int[] userRole = ChatTemplate.QWEN2.encodeRole(tokenizer, "user");
        int[] system = ChatTemplate.QWEN2.encodeSystemTurn(tokenizer, ChatTemplate.defaultSystemPromptText());
        int[] user = ChatTemplate.QWEN2.encodeUserTurn(tokenizer, "Hello");
        int[] prime = ChatTemplate.QWEN2.encodeAssistantPrime(tokenizer);
        int[] prompt = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "Hello");

        assertQwenRole(tokenizer, systemRole, "system", imStartId, newlineId);
        assertQwenEndTurn(tokenizer, system, imEndId, newlineId);

        assertQwenRole(tokenizer, userRole, "user", imStartId, newlineId);
        assertQwenEndTurn(tokenizer, user, imEndId, newlineId);

        assertQwenRole(tokenizer, prime, "assistant", imStartId, newlineId);
        assertTrue(prime.length == 3, "assistant prime is role header only");

        assertArrayEquals(
                ChatTemplate.concat(ChatTemplate.concat(system, user), prime),
                prompt,
                () -> tokenTrace(tokenizer, prompt));

        assertEquals(imStartId, prompt[0]);
        assertEquals(imStartId, prompt[system.length]);
        assertEquals(imStartId, prompt[system.length + user.length]);
    }

    @Test
    void llamaApplierTokenIdsMatchExistingChatTemplateEncoders() {
        ChatTokenizer tokenizer = qwenTokenizer();
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> messages = List.of(new ChatMessage("user", "Hello"));
        int[] fromApplier = applier.tokenIds(tokenizer, messages, true);
        int[] fromTemplate = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "Hello");
        assertArrayEquals(fromTemplate, fromApplier, () -> tokenTrace(tokenizer, fromApplier));
    }

    @Test
    void applierInjectsDefaultSystemFirstWhenMessagesOmitSystem() {
        ChatTokenizer tokenizer = qwenTokenizer();
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        int[] withUserOnly = applier.tokenIds(tokenizer, List.of(new ChatMessage("user", "Hello")), true);
        int[] expected = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "Hello");
        assertArrayEquals(expected, withUserOnly, () -> tokenTrace(tokenizer, withUserOnly));

        int[] systemTurn = ChatTemplate.QWEN2.encodeSystemTurn(tokenizer, ChatTemplate.defaultSystemPromptText());
        assertArrayEquals(systemTurn, java.util.Arrays.copyOfRange(withUserOnly, 0, systemTurn.length));
        assertEquals(tokenizer.tokenIdForText("system"), withUserOnly[1], "turn-1 prefill must open with system role");
    }

    @Test
    void generatorLlamaModeLeadsConversationWithSystemMessage() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);
        generator.continueConversation("Hello", ChatTemplate.QWEN2);

        List<ChatMessage> messages = generator.messages();
        assertTrue(messages.size() >= 2, "turn 1 must include system and user messages");
        assertEquals("system", messages.get(0).role());
        assertEquals(ChatTemplate.defaultSystemPromptText(), messages.get(0).content());
        assertEquals("user", messages.get(1).role());
        assertEquals("Hello", messages.get(1).content());

        ChatTokenizer tokenizer = model.tokenizer();
        int[] systemRole = ChatTemplate.QWEN2.encodeRole(tokenizer, "system");
        int[] session = generator.sessionTokenIds();
        assertArrayEquals(systemRole, java.util.Arrays.copyOfRange(session, 0, systemRole.length));
    }

    @Test
    void planPromptIdsMatchesEncodePromptForGenerationTurn1() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);
        int[] planned = generator.planPromptIds("Hello", ChatTemplate.QWEN2);
        int[] expected = ChatTemplate.QWEN2.encodePromptForGeneration(model.tokenizer(), "Hello");
        assertArrayEquals(expected, planned, () -> tokenTrace(model.tokenizer(), planned));
    }

    @Test
    void defaultSystemPromptFocusesNewestUserByDefault() {
        assertEquals(ChatTemplate.QWEN2_FOCUS_NEWEST_USER_SYSTEM, ChatTemplate.defaultSystemPromptText());
        assertEquals(ChatTemplate.QWEN2_FOCUS_NEWEST_USER_SYSTEM, ChatTemplate.parseSystemPromptMode(null));
        assertEquals(ChatTemplate.QWEN2_FOCUS_NEWEST_USER_SYSTEM, ChatTemplate.parseSystemPromptMode("focus"));
        assertEquals(ChatTemplate.QWEN2_CLASSIC_SYSTEM, ChatTemplate.parseSystemPromptMode("classic"));
    }

    @Test
    void defaultSystemTurnEnabledByDefault() {
        assertTrue(ChatTemplate.parseDefaultSystemTurnEnabled(null));
        assertTrue(ChatTemplate.parseDefaultSystemTurnEnabled(""));
        assertTrue(ChatTemplate.parseDefaultSystemTurnEnabled("true"));
        assertFalse(ChatTemplate.parseDefaultSystemTurnEnabled("false"));
    }

    @Test
    void turn1PromptDecodeHasSystemTurnBeforeUserTurn() {
        ChatTokenizer tokenizer = qwenTokenizer();
        int[] prompt = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "Hello");
        int[] systemTurn = ChatTemplate.QWEN2.encodeSystemTurn(tokenizer, ChatTemplate.defaultSystemPromptText());
        int[] userTurn = ChatTemplate.QWEN2.encodeUserTurn(tokenizer, "Hello");
        int[] assistantPrime = ChatTemplate.QWEN2.encodeAssistantPrime(tokenizer);

        assertArrayEquals(systemTurn, java.util.Arrays.copyOfRange(prompt, 0, systemTurn.length));
        assertArrayEquals(userTurn, java.util.Arrays.copyOfRange(prompt, systemTurn.length, systemTurn.length + userTurn.length));
        assertArrayEquals(
                assistantPrime,
                java.util.Arrays.copyOfRange(
                        prompt, systemTurn.length + userTurn.length, systemTurn.length + userTurn.length + assistantPrime.length));

        String decoded = tokenizer.decode(prompt);
        assertTrue(decoded.startsWith(IM_START + "system\n"), () -> "prefill must start with system ChatML turn: " + decoded);
        assertTrue(decoded.contains(IM_START + "user\nHello"), () -> "prefill must include user turn after system: " + decoded);
        assertTrue(decoded.endsWith(IM_START + "assistant\n"), () -> "prefill must end with assistant prime: " + decoded);
    }

    @Test
    void qwenTurnEndsWithEndTurnAndNewline() {
        ChatTokenizer tokenizer = qwenTokenizer();
        assertEquals(BpePreType.QWEN2, tokenizer.preType());

        int[] userTurn = ChatTemplate.QWEN2.encodeUserTurn(tokenizer, "Hello");
        int endTurn = tokenizer.endTurnId();
        int newlineId = newlineId(tokenizer);
        assertTrue(lastIndexOf(userTurn, endTurn) >= 0, "user turn must include im_end");
        assertEquals(newlineId, userTurn[userTurn.length - 1], "user turn must end with newline after im_end");
        assertEquals(tokenizer.tokenIdForText(IM_START), userTurn[0]);
    }

    @Test
    void firstTurnPrefillIsSystemThenUserThenAssistantPrime() {
        ChatTokenizer tokenizer = qwenTokenizer();
        int[] system = ChatTemplate.QWEN2.encodeSystemTurn(tokenizer, ChatTemplate.defaultSystemPromptText());
        int[] user = ChatTemplate.QWEN2.encodeUserTurn(tokenizer, "Hello");
        int[] prime = ChatTemplate.QWEN2.encodeAssistantPrime(tokenizer);
        int[] expected = ChatTemplate.concat(ChatTemplate.concat(system, user), prime);
        assertArrayEquals(expected, ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "Hello"));
    }

    @Test
    void promptForGenerationPrependsDefaultSystemWhenMissing() {
        ChatTokenizer tokenizer = qwenTokenizer();
        int[] systemPrefix = ChatTemplate.QWEN2.encodeDefaultSystemTurnIfMissing(tokenizer, false);
        int[] prompt = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "Hello");
        assertTrue(systemPrefix.length > 0);
        assertArrayEquals(systemPrefix, java.util.Arrays.copyOfRange(prompt, 0, systemPrefix.length));
    }

    @Test
    void encodeEndTurnAfterBodyAlreadyEndsWithImEndAppendsNewlineOnly() {
        ChatTokenizer tokenizer = qwenTokenizer();
        int imEnd = tokenizer.endTurnId();
        int[] suffix = ChatTemplate.QWEN2.encodeEndTurnAfter(tokenizer, new int[] {imEnd});
        assertArrayEquals(TinygradTokenizerReference.trailingNewlineAfterEndTurn(tokenizer), suffix);
        assertEquals(newlineId(tokenizer), suffix[0]);
    }

    @Test
    void fromTokenizerSelectsQwen2() {
        GgufFile file = MiniChatGgufBuilder.buildQwen2TemplateModel();
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(file.header());
        assertEquals(ChatTemplate.QWEN2, ChatTemplate.fromTokenizer(tokenizer));
    }

    private static ChatTokenizer qwenTokenizer() {
        return ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
    }

    private static int newlineId(ChatTokenizer tokenizer) {
        int[] encoded = tokenizer.encode("\n");
        if (encoded.length > 0) {
            return encoded[0];
        }
        return tokenizer.tokenIdForText("\n");
    }

    private static void assertQwenRole(
            ChatTokenizer tokenizer, int[] role, String roleName, int imStartId, int newlineId) {
        assertArrayEquals(TinygradTokenizerReference.role(tokenizer, roleName), role, roleName + " role");
        assertEquals(imStartId, role[0], roleName + " must open with im_start");
        assertEquals(tokenizer.tokenIdForText(roleName), role[1], roleName + " name must be one vocab token");
        assertEquals(newlineId, role[role.length - 1], roleName + " header must end with newline token");
    }

    private static void assertQwenEndTurn(
            ChatTokenizer tokenizer, int[] turn, int imEndId, int newlineId) {
        int imEndIndex = lastIndexOf(turn, imEndId);
        assertTrue(imEndIndex >= 0, () -> "turn must include im_end: " + tokenTrace(tokenizer, turn));
        assertEquals(newlineId, turn[imEndIndex + 1], "im_end must be followed by newline token");
        assertArrayEquals(
                TinygradTokenizerReference.endTurn(tokenizer, imEndId),
                java.util.Arrays.copyOfRange(turn, imEndIndex, turn.length),
                "turn suffix must match tinygrad end_turn");
    }

    /** Audit-style id legend for assertion failures. */
    private static String tokenTrace(ChatTokenizer tokenizer, int[] ids) {
        StringBuilder out = new StringBuilder();
        out.append("token trace len=").append(ids.length).append('\n');
        for (int id : ids) {
            out.append("  ").append(id).append(" -> ");
            try {
                out.append('"').append(tokenizer.tokenText(id)).append('"');
            } catch (IllegalArgumentException ex) {
                out.append("<out of range>");
            }
            out.append('\n');
        }
        try {
            out.append("decode: \"").append(tokenizer.decode(ids)).append('"');
        } catch (RuntimeException ex) {
            out.append("decode failed: ").append(ex.getMessage());
        }
        return out.toString();
    }

    private static int lastIndexOf(int[] array, int value) {
        for (int i = array.length - 1; i >= 0; i--) {
            if (array[i] == value) {
                return i;
            }
        }
        return -1;
    }
}
