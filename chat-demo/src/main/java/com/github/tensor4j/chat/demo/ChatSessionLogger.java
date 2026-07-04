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
import com.github.tensor4j.models.chat.ChatGenerator;
import com.github.tensor4j.models.chat.ChatTemplate;
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
            int eos = tokenizer.eosId();

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
                    "%d new tokens, %d prefix reused, session %d tokens, kv %d",
                    result.tokenCount(),
                    result.prefixReuseTokens(),
                    sessionAfter.length,
                    kvAfter));
            markdown.write("_\n\n");

            audit.write("=== turn ");
            audit.write(Integer.toString(turn));
            audit.write(" ===\n");
            audit.write("user_text: ");
            audit.write(userText);
            audit.write('\n');
            audit.write("user_turn_ids");
            audit.write(formatIds(userTurnIds));
            audit.write('\n');
            audit.write("assistant_prime_ids");
            audit.write(formatIds(assistantPrimeIds));
            audit.write('\n');
            audit.write("prompt_ids len=");
            audit.write(Integer.toString(promptIds.length));
            audit.write(formatIds(promptIds));
            audit.write('\n');
            audit.write("prompt_decode: ");
            audit.write(safeDecode(promptIds));
            audit.write('\n');
            audit.write("session_before len=");
            audit.write(Integer.toString(sessionTokensBefore.length));
            audit.write(formatIds(sessionTokensBefore));
            audit.write('\n');
            audit.write("cached_before len=");
            audit.write(Integer.toString(cachedTokensBefore.length));
            audit.write(formatIds(cachedTokensBefore));
            audit.write('\n');
            audit.write("kv_before=");
            audit.write(Integer.toString(kvBefore));
            audit.write(" kv_after=");
            audit.write(Integer.toString(kvAfter));
            audit.write(" prefix_reused=");
            audit.write(Integer.toString(result.prefixReuseTokens()));
            audit.write('\n');
            audit.write("generated_ids len=");
            audit.write(Integer.toString(result.generatedTokenIds().length));
            audit.write(formatIds(result.generatedTokenIds()));
            audit.write('\n');
            audit.write("forwarded_ids len=");
            audit.write(Integer.toString(result.forwardedTokenIds().length));
            audit.write(formatIds(result.forwardedTokenIds()));
            audit.write('\n');
            writeTokenLegend(audit, result.forwardedTokenIds());
            audit.write("session_after len=");
            audit.write(Integer.toString(sessionAfter.length));
            audit.write(" ends_eot=");
            audit.write(Boolean.toString(
                    sessionAfter.length > 0 && sessionAfter[sessionAfter.length - 1] == eos));
            audit.write(formatIds(sessionAfter));
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

    private void writeTokenLegend(BufferedWriter out, int[] ids) throws IOException {
        out.write("forwarded_token_map:\n");
        for (int id : ids) {
            out.write("  ");
            out.write(Integer.toString(id));
            out.write(" -> ");
            out.write(describeToken(id));
            out.write('\n');
        }
    }

    private String describeToken(int id) {
        if (id == tokenizer.bosId()) {
            return "<|bos|>";
        }
        if (id == tokenizer.eosId()) {
            return "<|eot_id|>";
        }
        if (tokenizer.skipGeneratedPiece(id)) {
            try {
                return tokenizer.tokenText(id);
            } catch (IllegalArgumentException ex) {
                return "<|special|>";
            }
        }
        String piece = tokenizer.tryDecodePiece(id);
        if (piece == null) {
            try {
                return tokenizer.tokenText(id);
            } catch (IllegalArgumentException ex) {
                return "<|unknown|>";
            }
        }
        return quote(piece);
    }

    private static String quote(String text) {
        return "\"" + text.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
    }

    private String safeDecode(int[] ids) {
        try {
            return quote(tokenizer.decode(ids));
        } catch (IllegalArgumentException ex) {
            return "(decode failed: " + ex.getMessage() + ")";
        }
    }

    private static String formatIds(int[] ids) {
        if (ids.length == 0) {
            return " []";
        }
        StringBuilder out = new StringBuilder(" [");
        int show = Math.min(ids.length, 512);
        for (int i = 0; i < show; i++) {
            if (i > 0) {
                out.append(", ");
            }
            out.append(ids[i]);
        }
        if (ids.length > show) {
            out.append(", ... (+");
            out.append(ids.length - show);
            out.append(" more)");
        }
        out.append(']');
        return out.toString();
    }

    private void flush() throws IOException {
        markdown.flush();
        audit.flush();
    }
}
