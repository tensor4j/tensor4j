/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.models.chat.ChatTokenizer;
import java.util.ArrayList;
import java.util.List;

/** Top-k logits report for prefill parity debugging (tinygrad vs tensor4j). */
public final class LogitsTopK {

    public record Entry(int tokenId, float logit, String piece) {
    }

    private LogitsTopK() {
    }

    public static List<Entry> topK(float[] logits, ChatTokenizer tokenizer, int k) {
        if (k <= 0 || logits.length == 0) {
            return List.of();
        }
        int limit = Math.min(k, logits.length);
        int[] order = new int[logits.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        for (int i = 0; i < limit; i++) {
            int best = i;
            for (int j = i + 1; j < order.length; j++) {
                if (logits[order[j]] > logits[order[best]]) {
                    best = j;
                }
            }
            int tmp = order[i];
            order[i] = order[best];
            order[best] = tmp;
        }
        List<Entry> entries = new ArrayList<>(limit);
        for (int i = 0; i < limit; i++) {
            int id = order[i];
            entries.add(new Entry(id, logits[id], safeDecode(tokenizer, id)));
        }
        return entries;
    }

    private static String safeDecode(ChatTokenizer tokenizer, int tokenId) {
        try {
            return tokenizer.decode(new int[] {tokenId});
        } catch (RuntimeException ex) {
            return "<id=" + tokenId + ">";
        }
    }

    public static String formatReport(String label, float[] logits, ChatTokenizer tokenizer, int k) {
        StringBuilder sb = new StringBuilder();
        sb.append(label).append('\n');
        int rank = 1;
        for (Entry entry : topK(logits, tokenizer, k)) {
            sb.append(String.format("%2d. %-20s  id=%6d  logit=% .5f%n",
                    rank++, quote(entry.piece()), entry.tokenId(), entry.logit()));
        }
        return sb.toString();
    }

    public static boolean topKMatches(
            float[] expected,
            float[] actual,
            ChatTokenizer expectedTok,
            ChatTokenizer actualTok,
            int k,
            float logitTolerance) {
        List<Entry> exp = topK(expected, expectedTok, k);
        List<Entry> act = topK(actual, actualTok, k);
        if (exp.size() != act.size()) {
            return false;
        }
        for (int i = 0; i < exp.size(); i++) {
            if (exp.get(i).tokenId() != act.get(i).tokenId()) {
                return false;
            }
            if (Math.abs(exp.get(i).logit() - act.get(i).logit()) > logitTolerance) {
                return false;
            }
        }
        return true;
    }

    private static String quote(String piece) {
        if (piece == null) {
            return "null";
        }
        return "\"" + piece.replace("\n", "\\n").replace("\r", "\\r") + "\"";
    }
}
