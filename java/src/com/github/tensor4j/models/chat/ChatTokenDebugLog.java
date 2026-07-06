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

import java.io.IOException;
import java.io.PrintStream;
import java.io.Writer;
import java.util.Locale;

/** Wrapped token-id traces with vocab pieces ({@code TENSOR4J_CHAT_DEBUG}, default on). */
public final class ChatTokenDebugLog {

    private static final int DEFAULT_LINE_WIDTH = 100;

    private ChatTokenDebugLog() {}

    /** Default {@code true}; set {@code TENSOR4J_CHAT_DEBUG=false} to silence stderr token traces. */
    public static boolean enabled() {
        return parseEnabled(System.getenv("TENSOR4J_CHAT_DEBUG"));
    }

    static boolean parseEnabled(String raw) {
        if (raw == null || raw.isBlank()) {
            return true;
        }
        return Boolean.parseBoolean(raw.trim());
    }

    public static int lineWidth() {
        String raw = System.getenv("TENSOR4J_CHAT_TOKEN_DEBUG_WIDTH");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_LINE_WIDTH;
        }
        try {
            return Math.max(40, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException ex) {
            return DEFAULT_LINE_WIDTH;
        }
    }

    public static void log(PrintStream out, String label, ChatTokenizer tokenizer, int[] ids) {
        if (!enabled()) {
            return;
        }
        try {
            out.println("--- " + label + " ---");
            writeBlock(out, label, tokenizer, ids);
            out.println("--- end " + label + " ---");
        } catch (IOException ex) {
            out.printf(Locale.ROOT, "%s: (debug log failed: %s)%n", label, ex.getMessage());
        }
    }

    public static String promptDecodeLabel(int templatePrevTokens, boolean deltaAppend) {
        if (!deltaAppend) {
            return "prompt_decode_prefill";
        }
        return templatePrevTokens == 0 ? "prompt_decode_turn1" : "prompt_decode_delta";
    }

    /**
     * Full contiguous token vector after prefill: {@code session_tokens + prefill_this_turn}.
     * Assistant {@code im_end} must sit at {@code [sessionLen-2:sessionLen]} before user {@code im_start} at
     * {@code [sessionLen]}.
     */
    public static void logModelInputTokens(
            PrintStream out, ChatTokenizer tokenizer, int[] modelInput, int sessionLen) {
        if (!enabled()) {
            return;
        }
        log(out, "model_input_tokens", tokenizer, modelInput);
        if (sessionLen <= 0 || modelInput.length <= sessionLen) {
            return;
        }
        int endTurn = tokenizer.endTurnId();
        int newlineId = newlineId(tokenizer);
        int imStart = tokenizer.tokenIdForText("<|im_start|>");
        out.println("--- model_input_boundary ---");
        out.printf(
                Locale.ROOT,
                "  [%d]=%d (%s) [%d]=%d (newline) -> [%d]=%d (user im_start)%n",
                sessionLen - 2,
                modelInput[sessionLen - 2],
                describePiece(tokenizer, modelInput[sessionLen - 2]),
                sessionLen - 1,
                modelInput[sessionLen - 1],
                sessionLen,
                modelInput[sessionLen]);
        boolean ok = modelInput[sessionLen - 2] == endTurn
                && modelInput[sessionLen - 1] == newlineId
                && modelInput[sessionLen] == imStart;
        out.printf(Locale.ROOT, "  assistant_im_end_before_user_im_start: %s%n", ok ? "ok" : "MISSING");
        if (!ok) {
            out.println("  ERROR: model input token array missing assistant im_end before second user turn");
        }
        out.println("--- end model_input_boundary ---");
    }

    public static void logPromptText(PrintStream out, String label, String promptText) {
        if (!enabled()) {
            return;
        }
        out.println("--- " + label + " ---");
        out.println(promptText);
        out.println("--- end " + label + " ---");
    }

    /** After turn close or before turn N+1 user input — KV vs closed-template token count. */
    public static void logKvHandoff(PrintStream out, int kvLength, int templatePrevTokens, int[] suffixForwarded) {
        if (!enabled()) {
            return;
        }
        out.println("--- turn_handoff ---");
        out.printf(
                Locale.ROOT,
                "  kv_length=%d template_prev_tokens=%d synced=%s%n",
                kvLength,
                templatePrevTokens,
                kvLength == templatePrevTokens);
        if (suffixForwarded != null && suffixForwarded.length > 0) {
            out.print("  turn_close_suffix ");
            out.println(formatIds(suffixForwarded));
        }
        out.println("--- end turn_handoff ---");
    }

    /**
     * Explains delta KV: prior turns stay in KV; only {@code full[templatePrevTokens:]} is prefilled this turn.
     */
    public static void logPromptDecodeContext(
            PrintStream out,
            ChatTokenizer tokenizer,
            int kvBefore,
            int templatePrevTokens,
            int[] closedSession,
            int[] fullPrompt,
            int[] deltaPrompt) {
        if (!enabled() || templatePrevTokens <= 0) {
            return;
        }
        boolean deltaMatchesSuffix = fullPrompt.length >= templatePrevTokens
                && deltaPrompt.length == fullPrompt.length - templatePrevTokens;
        if (deltaMatchesSuffix) {
            for (int i = 0; i < deltaPrompt.length; i++) {
                if (fullPrompt[templatePrevTokens + i] != deltaPrompt[i]) {
                    deltaMatchesSuffix = false;
                    break;
                }
            }
        }
        out.println("--- prompt_decode_context ---");
        out.printf(
                Locale.ROOT,
                "  prior_turns_in_kv=%d tokens (in KV cache — see closed_session_before_user; not in delta)%n",
                templatePrevTokens);
        out.printf(
                Locale.ROOT,
                "  kv_before=%d synced_with_template_prev=%s%n",
                kvBefore,
                kvBefore == templatePrevTokens);
        out.printf(
                Locale.ROOT,
                "  delta_prefill_this_turn=%d tokens (prompt_decode_delta — new user turn + assistant prime)%n",
                deltaPrompt.length);
        out.printf(Locale.ROOT, "  full_cold_template=%d tokens (prompt_decode_full — all turns if cache were empty)%n", fullPrompt.length);
        out.printf(
                Locale.ROOT,
                "  model_sees=%d tokens total after prefill (kv_before + delta)%n",
                kvBefore + deltaPrompt.length);
        out.printf(Locale.ROOT, "  delta_equals_full_suffix[%d:]: %s%n", templatePrevTokens, deltaMatchesSuffix);
        logDeltaBoundary(out, tokenizer, closedSession, fullPrompt, templatePrevTokens, deltaPrompt);
        out.println("--- end prompt_decode_context ---");
    }

    /**
     * Shows the last tokens before {@code prompt_decode_delta} — turn-1 assistant {@code im_end} lives here,
     * not inside the delta slice.
     */
    public static void logDeltaBoundary(
            PrintStream out,
            ChatTokenizer tokenizer,
            int[] closedSession,
            int[] fullPrompt,
            int templatePrevTokens,
            int[] deltaPrompt) {
        if (!enabled() || templatePrevTokens <= 0) {
            return;
        }
        int tailLen = Math.min(8, templatePrevTokens);
        int[] prefixTail = java.util.Arrays.copyOfRange(fullPrompt, templatePrevTokens - tailLen, templatePrevTokens);
        out.println("  --- prior_kv_tail (full[" + (templatePrevTokens - tailLen) + ":" + templatePrevTokens + "]; not in delta) ---");
        out.println("  " + formatIds(prefixTail));
        for (int id : prefixTail) {
            out.printf(Locale.ROOT, "  %d %s%n", id, describePiece(tokenizer, id));
        }
        int endTurn = tokenizer.endTurnId();
        int newlineId = newlineId(tokenizer);
        boolean fullBoundaryOk =
                templatePrevTokens >= 2
                        && fullPrompt[templatePrevTokens - 2] == endTurn
                        && fullPrompt[templatePrevTokens - 1] == newlineId;
        boolean sessionBoundaryOk =
                closedSession.length >= 2
                        && closedSession[closedSession.length - 2] == endTurn
                        && closedSession[closedSession.length - 1] == newlineId;
        out.printf(
                Locale.ROOT,
                "  turn1_assistant_im_end_before_delta: full_prefix=%s closed_session=%s%n",
                fullBoundaryOk,
                sessionBoundaryOk);
        if (!fullBoundaryOk || !sessionBoundaryOk) {
            out.println("  WARN: expected ... im_end, newline immediately before delta user turn");
        }
        if (deltaPrompt.length >= 3) {
            out.printf(
                    Locale.ROOT,
                    "  delta_starts_with: [%d,%d,%d] (%s user header expected)%n",
                    deltaPrompt[0],
                    deltaPrompt[1],
                    deltaPrompt[2],
                    deltaPrompt[0] == tokenizer.tokenIdForText("<|im_start|>") ? "ok" : "unexpected");
        }
        out.println("  --- end prior_kv_tail ---");
    }

    private static int newlineId(ChatTokenizer tokenizer) {
        int[] encoded = tokenizer.encode("\n");
        return encoded.length > 0 ? encoded[0] : tokenizer.tokenIdForText("\n");
    }

    /** Per-token decode trace after the assistant reply (audit steps, not live stderr during stream). */
    public static void logGenerationSteps(PrintStream out, ChatTokenizer tokenizer, ChatGenerationStep[] steps) {
        if (!enabled() || steps == null || steps.length == 0) {
            return;
        }
        out.println("--- generation_steps ---");
        for (ChatGenerationStep step : steps) {
            String piece = step.piece();
            if (piece != null && step.visible()) {
                out.printf(
                        Locale.ROOT,
                        "  step %d: id=%d text=%s%n",
                        step.step(),
                        step.tokenId(),
                        quote(piece));
            } else {
                out.printf(
                        Locale.ROOT,
                        "  step %d: id=%d%s%n",
                        step.step(),
                        step.tokenId(),
                        describePiece(tokenizer, step.tokenId()));
            }
        }
        out.println("--- end generation_steps ---");
    }

    public static void writeBlock(Writer out, String label, ChatTokenizer tokenizer, int[] ids) throws IOException {
        out.write(label);
        out.write(" len=");
        out.write(Integer.toString(ids.length));
        out.write('\n');
        out.write(formatIds(ids));
        out.write('\n');
        out.write(formatTokenMap(tokenizer, ids));
    }

    private static void writeBlock(PrintStream out, String label, ChatTokenizer tokenizer, int[] ids)
            throws IOException {
        AppendableWriter wrapper = new AppendableWriter(out);
        writeBlock(wrapper, label, tokenizer, ids);
    }

    public static String formatIds(int[] ids) {
        if (ids.length == 0) {
            return "[]";
        }
        int width = lineWidth();
        StringBuilder line = new StringBuilder("[");
        boolean firstOnLine = true;
        for (int i = 0; i < ids.length; i++) {
            String token = Integer.toString(ids[i]);
            String sep = firstOnLine ? "" : ", ";
            if (!firstOnLine && line.length() + sep.length() + token.length() + 1 > width) {
                line.append(",\n ");
                sep = "";
                firstOnLine = true;
            }
            line.append(sep).append(token);
            firstOnLine = false;
        }
        line.append(']');
        return line.toString();
    }

    public static String formatTokenMap(ChatTokenizer tokenizer, int[] ids) {
        if (ids.length == 0) {
            return "";
        }
        int width = lineWidth();
        StringBuilder out = new StringBuilder();
        for (int id : ids) {
            String entry = id + " " + describePiece(tokenizer, id);
            if (out.length() > 0) {
                out.append('\n');
            }
            appendWrappedLine(out, "  ", entry, width);
        }
        return out.toString();
    }

    public static String describePiece(ChatTokenizer tokenizer, int id) {
        if (id == tokenizer.bosId()) {
            return "-> <|bos|>";
        }
        if (id == tokenizer.endTurnId()) {
            return "-> " + quote(safeTokenText(tokenizer, id));
        }
        if (id == tokenizer.eosId() && id != tokenizer.endTurnId()) {
            return "-> " + quote(safeTokenText(tokenizer, id));
        }
        if (id == tokenizer.eotId() && id != tokenizer.endTurnId()) {
            return "-> " + quote(safeTokenText(tokenizer, id));
        }
        if (tokenizer.skipGeneratedPiece(id)) {
            try {
                return "-> " + quote(tokenizer.tokenText(id));
            } catch (IllegalArgumentException ex) {
                return "-> <|special|>";
            }
        }
        String piece = tokenizer.tryDecodePiece(id);
        if (piece == null) {
            try {
                return "-> " + quote(tokenizer.tokenText(id));
            } catch (IllegalArgumentException ex) {
                return "-> <|unknown|>";
            }
        }
        return "-> " + quote(piece);
    }

    private static void appendWrappedLine(StringBuilder out, String indent, String text, int width) {
        int pos = 0;
        boolean first = true;
        while (pos < text.length()) {
            if (!first) {
                out.append('\n').append(indent);
            } else {
                out.append(indent);
                first = false;
            }
            int chunk = Math.min(text.length() - pos, width - indent.length());
            out.append(text, pos, pos + chunk);
            pos += chunk;
        }
    }

    public static String quote(String text) {
        return "\"" + text.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
    }

    private static String safeTokenText(ChatTokenizer tokenizer, int id) {
        try {
            return tokenizer.tokenText(id);
        } catch (IllegalArgumentException ex) {
            return "<|unknown|>";
        }
    }

    private static final class AppendableWriter extends Writer {
        private final Appendable target;

        AppendableWriter(Appendable target) {
            this.target = target;
        }

        @Override
        public void write(char[] cbuf, int off, int len) throws IOException {
            target.append(new String(cbuf, off, len));
        }

        @Override
        public void flush() {}

        @Override
        public void close() {}
    }
}
