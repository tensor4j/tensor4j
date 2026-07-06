/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.chat.demo;

import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatGenerationResult;
import com.github.tensor4j.models.chat.ChatGenerationStep;
import com.github.tensor4j.models.chat.ChatGenerationStopReason;
import com.github.tensor4j.models.chat.ChatGenerator;
import com.github.tensor4j.models.chat.ChatMessage;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.ChatTokenDebugLog;
import com.github.tensor4j.models.chat.ChatTokenizer;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Locale;

/** Incremental dual transcript: readable markdown + verbose token audit. */
final class ChatSessionLogger implements AutoCloseable {

    private static final DateTimeFormatter TITLE_STAMP =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss", Locale.ROOT);

    private final Path markdownPath;
    private final Path auditPath;
    private final BufferedWriter markdown;
    private final BufferedWriter audit;
    private final ChatTokenizer tokenizer;
    private int turnCount;

    ChatSessionLogger(
            Path saveDir,
            String fileBase,
            String ggufPath,
            ChatTemplate template,
            ChatGenerationOptions options,
            ChatTokenizer tokenizer)
            throws IOException {
        this.tokenizer = tokenizer;
        this.markdownPath = saveDir.resolve(fileBase + "-chat.md");
        this.auditPath = saveDir.resolve(fileBase + "-chat-audit.log");
        this.markdown = Files.newBufferedWriter(markdownPath, StandardCharsets.UTF_8);
        this.audit = Files.newBufferedWriter(auditPath, StandardCharsets.UTF_8);
        LocalDateTime now = LocalDateTime.now();
        writeMarkdownHeader(now, ggufPath, template, options);
        writeAuditHeader(now, ggufPath, template, options);
        flush();
    }

    Path markdownPath() {
        return markdownPath;
    }

    Path auditPath() {
        return auditPath;
    }

    void logSystem(String message) {
        try {
            markdown.write("## System\n\n");
            markdown.write(message);
            markdown.write("\n\n");
            audit.write("=== system ===\n");
            audit.write(message);
            audit.write('\n');
            audit.write('\n');
            flush();
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    void logTurn(
            int turn,
            String userText,
            int[] systemTurnIds,
            int[] userTurnIds,
            int[] assistantPrimeIds,
            int[] promptIds,
            int[] sessionTokensBefore,
            int[] cachedTokensBefore,
            int kvBefore,
            ChatGenerationResult result,
            ChatGenerator generator) {
        try {
            int[] sessionAfter = generator.sessionTokenIds();
            int kvAfter = generator.kvLength();
            int endTurn = tokenizer.endTurnId();
            int newlineAfterEndTurn = newlineTokenId();

            markdown.write("## User\n\n");
            markdown.write(userText.isBlank() ? "_(empty)_" : userText);
            markdown.write("\n\n");
            markdown.write("## Assistant\n\n");
            String reply = result.text();
            markdown.write(reply.isBlank() ? "_(empty)_" : reply);
            markdown.write("\n\n");
            markdown.write("_");
            markdown.write(String.format(
                    Locale.US,
                    "%d new tokens, stop=%s%s, %d left, %s, session %d tokens, kv %d",
                    result.tokenCount(),
                    result.stopReason().name().toLowerCase(Locale.ROOT),
                    result.stopTokenId() >= 0 ? " id=" + result.stopTokenId() : "",
                    result.tokensRemaining(),
                    result.prefixReuseTokens() == 0 ? "delta-only" : result.prefixReuseTokens() + " prefix reused",
                    sessionAfter.length,
                    kvAfter));
            markdown.write("_\n\n");

            audit.write("=== turn ");
            audit.write(Integer.toString(turn));
            audit.write(" ===\n");
            audit.write("user_text: ");
            audit.write(userText);
            audit.write('\n');
            writeMessagesHistory(audit, generator);
            writeTokenBlock(audit, "system_turn_ids", systemTurnIds);
            writeTokenBlock(audit, "user_turn_ids", userTurnIds);
            writeTokenBlock(audit, "assistant_prime_ids", assistantPrimeIds);
            writeTokenBlock(audit, "prompt_ids", promptIds);
            audit.write("prompt_decode: ");
            audit.write(safeDecode(promptIds));
            audit.write('\n');
            writeTokenBlock(audit, "session_before", sessionTokensBefore);
            writeTokenBlock(audit, "cached_before", cachedTokensBefore);
            audit.write("kv_before=");
            audit.write(Integer.toString(kvBefore));
            audit.write(" kv_after=");
            audit.write(Integer.toString(kvAfter));
            audit.write(" prefix_reused=");
            audit.write(Integer.toString(result.prefixReuseTokens()));
            audit.write(" stop_reason=");
            audit.write(result.stopReason().name());
            audit.write(" stop_token_id=");
            audit.write(Integer.toString(result.stopTokenId()));
            audit.write(" tokens_remaining=");
            audit.write(Integer.toString(result.tokensRemaining()));
            audit.write('\n');
            writeGenerationSteps(audit, result);
            writeTokenBlock(audit, "generated_ids", result.generatedTokenIds());
            writeTokenBlock(audit, "forwarded_ids", result.forwardedTokenIds());
            writeTokenBlock(audit, "session_after", sessionAfter);
            audit.write("session_tail");
            audit.write(formatTail(sessionAfter, 6));
            audit.write('\n');
            audit.write("session_ends_im_end_newline: ");
            audit.write(Boolean.toString(endsWithImEndNewline(sessionAfter, endTurn, newlineAfterEndTurn)));
            audit.write('\n');
            audit.write("session_decode: ");
            audit.write(safeDecode(sessionAfter));
            audit.write('\n');
            int[] cachedAfter = generator.cachedTokenIds();
            audit.write("cached_equals_session: ");
            audit.write(Boolean.toString(Arrays.equals(cachedAfter, sessionAfter)));
            audit.write('\n');
            audit.write("kv_equals_session_len: ");
            audit.write(Boolean.toString(kvAfter == sessionAfter.length));
            audit.write(" kv_equals_template_prev: ");
            audit.write(Boolean.toString(kvAfter == generator.templatePrevTokens()));
            audit.write('\n');
            audit.write("prompt_extends_session_before: ");
            audit.write(Boolean.toString(prefixMatches(sessionTokensBefore, promptIds)));
            audit.write('\n');
            audit.write("kv_reuse_enabled: ");
            audit.write(Boolean.toString(ChatGenerator.kvReuseEnabled()));
            audit.write('\n');
            audit.write('\n');
            flush();
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    int nextTurnNumber() {
        return ++turnCount;
    }

    @Override
    public void close() throws IOException {
        markdown.close();
        audit.close();
    }

    private void writeMarkdownHeader(
            LocalDateTime now, String ggufPath, ChatTemplate template, ChatGenerationOptions options)
            throws IOException {
        markdown.write("# Chat — ");
        markdown.write(TITLE_STAMP.format(now));
        markdown.write("\n\n");
        markdown.write("- **Model:** `");
        markdown.write(ggufPath);
        markdown.write("`\n");
        markdown.write("- **Template:** ");
        markdown.write(template.name().toLowerCase(Locale.ROOT));
        markdown.write('\n');
        markdown.write("- **Sampling:** ");
        markdown.write(options.mode().name().toLowerCase(Locale.ROOT));
        markdown.write(" (max ");
        markdown.write(Integer.toString(options.maxNewTokens()));
        markdown.write(" new tokens / reply)\n\n");
    }

    private void writeAuditHeader(
            LocalDateTime now, String ggufPath, ChatTemplate template, ChatGenerationOptions options)
            throws IOException {
        audit.write("# tensor4j chat audit — ");
        audit.write(TITLE_STAMP.format(now));
        audit.write('\n');
        audit.write("model: ");
        audit.write(ggufPath);
        audit.write('\n');
        audit.write("template: ");
        audit.write(template.name());
        audit.write('\n');
        audit.write("mode: ");
        audit.write(options.mode().name());
        audit.write(" max_new_tokens=");
        audit.write(Integer.toString(options.maxNewTokens()));
        audit.write(" min_new_tokens=");
        audit.write(Integer.toString(options.minNewTokens()));
        audit.write(" temperature=");
        audit.write(Float.toString(options.temperature()));
        audit.write(" top_p=");
        audit.write(Float.toString(options.topP()));
        audit.write(" top_k=");
        audit.write(Integer.toString(options.topK()));
        audit.write(" alpha_f=");
        audit.write(Float.toString(options.alphaFrequency()));
        audit.write(" alpha_p=");
        audit.write(Float.toString(options.alphaPresence()));
        audit.write('\n');
        audit.write("bos_id=");
        audit.write(Integer.toString(options.bosId()));
        audit.write(" eos_id=");
        audit.write(Integer.toString(options.eosId()));
        audit.write(" eot_id=");
        audit.write(Integer.toString(options.eotId()));
        audit.write(" end_turn_id=");
        audit.write(Integer.toString(tokenizer.endTurnId()));
        audit.write(" default_system=");
        audit.write(Boolean.toString(ChatTemplate.defaultSystemTurnEnabled()));
        audit.write(" kv_reuse=");
        audit.write(Boolean.toString(ChatGenerator.kvReuseEnabled()));
        audit.write('\n');
        audit.write('\n');
    }

    private static boolean prefixMatches(int[] prefix, int[] full) {
        if (prefix.length > full.length) {
            return false;
        }
        for (int i = 0; i < prefix.length; i++) {
            if (prefix[i] != full[i]) {
                return false;
            }
        }
        return true;
    }

    private void writeGenerationSteps(BufferedWriter out, ChatGenerationResult result) throws IOException {
        ChatGenerationStep[] steps = result.steps();
        if (steps.length == 0) {
            return;
        }
        out.write("generation_steps:\n");
        for (ChatGenerationStep step : steps) {
            out.write("  step ");
            out.write(Integer.toString(step.step()));
            out.write(": id=");
            out.write(Integer.toString(step.tokenId()));
            out.write(" visible=");
            out.write(Boolean.toString(step.visible()));
            out.write(" eog=");
            out.write(Boolean.toString(step.endOfGeneration()));
            out.write(" text=");
            out.write(ChatTokenDebugLog.describePiece(tokenizer, step.tokenId()));
            out.write('\n');
        }
    }

    private void writeTokenBlock(BufferedWriter out, String label, int[] ids) throws IOException {
        ChatTokenDebugLog.writeBlock(out, label, tokenizer, ids);
        out.write('\n');
    }

    private void writeMessagesHistory(BufferedWriter out, ChatGenerator generator) throws IOException {
        out.write("messages_history:\n");
        int i = 0;
        for (ChatMessage message : generator.messages()) {
            out.write("  [");
            out.write(Integer.toString(i++));
            out.write("] ");
            out.write(message.role());
            out.write(": ");
            out.write(message.content().replace("\n", "\\n"));
            if (message.generatedTokenIds() != null) {
                out.write(" (sampled_ids len=");
                out.write(Integer.toString(message.generatedTokenIds().length));
                out.write(')');
            }
            out.write('\n');
        }
    }

    private int newlineTokenId() {
        int[] encoded = tokenizer.encode("\n");
        return encoded.length > 0 ? encoded[0] : -1;
    }

    private static boolean endsWithImEndNewline(int[] session, int imEndId, int newlineId) {
        if (session.length < 2 || newlineId < 0) {
            return false;
        }
        return session[session.length - 1] == newlineId && session[session.length - 2] == imEndId;
    }

    private String formatTail(int[] ids, int count) {
        if (ids.length == 0) {
            return " []";
        }
        int start = Math.max(0, ids.length - count);
        int[] tail = Arrays.copyOfRange(ids, start, ids.length);
        return ChatTokenDebugLog.formatIds(tail) + "\n" + ChatTokenDebugLog.formatTokenMap(tokenizer, tail);
    }

    private String safeDecode(int[] ids) {
        try {
            return ChatTokenDebugLog.quote(tokenizer.decode(ids));
        } catch (IllegalArgumentException ex) {
            return "(decode failed: " + ex.getMessage() + ")";
        }
    }

    private void flush() throws IOException {
        markdown.flush();
        audit.flush();
    }
}
