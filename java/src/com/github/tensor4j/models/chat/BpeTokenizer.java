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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Byte-pair encoding tokenizer ({@code llm_tokenizer_bpe_session} in llama-vocab.cpp).
 * Supports GPT-2 byte encoding, default pre-split regex, and whole-word vocab lookup before merges.
 */
final class BpeTokenizer {

    private final String[] tokens;
    private final Map<String, Integer> tokenToId;
    private final Map<String, Integer> mergeRanks;
    private final BpePreType preType;
    private final BpeSplitMode splitMode;
    private final boolean ignoreMerges;

    BpeTokenizer(String[] tokens, String[] merges, BpePreType preType, BpeSplitMode splitMode, boolean ignoreMerges) {
        this.tokens = tokens.clone();
        this.tokenToId = new HashMap<>();
        for (int i = 0; i < tokens.length; i++) {
            tokenToId.put(tokens[i], i);
        }
        this.mergeRanks = buildMergeRanks(merges);
        this.preType = preType;
        this.splitMode = splitMode;
        this.ignoreMerges = ignoreMerges;
    }

    int[] encode(String text) {
        List<Integer> out = new ArrayList<>();
        List<String> words = splitWords(text);
        for (String word : words) {
            encodeWord(word, out);
        }
        int[] ids = new int[out.size()];
        for (int i = 0; i < out.size(); i++) {
            ids[i] = out.get(i);
        }
        return ids;
    }

    String decode(int[] ids) {
        StringBuilder out = new StringBuilder();
        for (int id : ids) {
            if (id < 0 || id >= tokens.length) {
                throw new IllegalArgumentException("token id out of range " + id);
            }
            out.append(tokens[id]);
        }
        if (preType.byteEncode()) {
            return Gpt2ByteEncoder.decode(out.toString());
        }
        return out.toString();
    }

    private List<String> splitWords(String text) {
        String input = preType.byteEncode() ? Gpt2ByteEncoder.encode(text) : text;
        if (preType == BpePreType.WHITESPACE) {
            return splitWhitespaceSession(input);
        }
        List<String> words = BpePreSplit.split(input, preType, splitMode);
        if (words.isEmpty() && !input.isEmpty()) {
            words = List.of(input);
        }
        return words;
    }

    /** {@code llm_tokenizer_whitespace_session} — flush on whitespace, then BPE each segment. */
    private List<String> splitWhitespaceSession(String text) {
        int[] cpts = com.github.tensor4j.runtime.unicode.UnicodeCpt.codepointsFromUtf8(text);
        List<String> segments = new ArrayList<>();
        StringBuilder segment = new StringBuilder();
        for (int cpt : cpts) {
            if (com.github.tensor4j.runtime.unicode.UnicodeCptFlags.fromCodepoint(cpt).isWhitespace()) {
                flushWhitespaceSegment(segments, segment);
            } else {
                segment.appendCodePoint(cpt);
            }
        }
        flushWhitespaceSegment(segments, segment);
        List<String> words = new ArrayList<>();
        for (String part : segments) {
            words.addAll(BpePreSplit.split(part, preType, splitMode));
        }
        return words;
    }

    private static void flushWhitespaceSegment(List<String> segments, StringBuilder segment) {
        if (segment.length() > 0) {
            segments.add(segment.toString());
            segment.setLength(0);
        }
    }

    private void encodeWord(String word, List<Integer> out) {
        if (preType == BpePreType.WHITESPACE && tokenToId.containsKey(word)) {
            out.add(tokenToId.get(word));
            return;
        }
        if (ignoreMerges && tokenToId.containsKey(word)) {
            out.add(tokenToId.get(word));
            return;
        }
        if (preType == BpePreType.GEMMA4 && isNewlineOnly(word)) {
            Integer id = tokenToId.get(word);
            if (id != null) {
                out.add(id);
                return;
            }
        }

        List<Symbol> symbols = new ArrayList<>();
        int index = 0;
        int offset = 0;
        while (offset < word.length()) {
            int charLen = utf8CharLength(word, offset);
            String piece = word.substring(offset, offset + charLen);
            symbols.add(new Symbol(piece, index - 1, offset + charLen >= word.length() ? -1 : index + 1));
            offset += charLen;
            index++;
        }

        PriorityBigramQueue queue = new PriorityBigramQueue();
        for (int i = 1; i < symbols.size(); i++) {
            addBigram(symbols, queue, i - 1, i);
        }

        while (!queue.isEmpty()) {
            Bigram bigram = queue.pop();
            Symbol left = symbols.get(bigram.left);
            Symbol right = symbols.get(bigram.right);
            if (left.empty || right.empty) {
                continue;
            }
            if (!left.text.equals(bigram.leftText) || !right.text.equals(bigram.rightText)) {
                continue;
            }
            left.text = left.text + right.text;
            right.empty = true;
            left.next = right.next;
            if (right.next >= 0) {
                symbols.get(right.next).prev = bigram.left;
            }
            addBigram(symbols, queue, left.prev, bigram.left);
            addBigram(symbols, queue, bigram.left, left.next);
        }

        for (int i = 0; i != -1; i = nextIndex(symbols, i)) {
            Symbol sym = symbols.get(i);
            if (sym.empty) {
                continue;
            }
            Integer id = tokenToId.get(sym.text);
            if (id != null) {
                out.add(id);
            } else {
                emitFallbackBytes(sym.text, out);
            }
        }
    }

    private void emitFallbackBytes(String text, List<Integer> out) {
        for (int i = 0; i < text.length(); i += utf8CharLength(text, i)) {
            String piece = text.substring(i, i + utf8CharLength(text, i));
            Integer id = tokenToId.get(piece);
            if (id != null) {
                out.add(id);
            }
        }
    }

    private void addBigram(List<Symbol> symbols, PriorityBigramQueue queue, int left, int right) {
        if (left < 0 || right < 0) {
            return;
        }
        Symbol leftSym = symbols.get(left);
        Symbol rightSym = symbols.get(right);
        if (leftSym.empty || rightSym.empty) {
            return;
        }
        Integer rank = mergeRanks.get(leftSym.text + rightSym.text);
        if (rank == null) {
            return;
        }
        queue.push(new Bigram(left, right, leftSym.text, rightSym.text, rank));
    }

    private static int nextIndex(List<Symbol> symbols, int index) {
        if (index < 0) {
            return -1;
        }
        return symbols.get(index).next;
    }

    private static int utf8CharLength(String text, int offset) {
        if (offset >= text.length()) {
            return 0;
        }
        if (Character.isHighSurrogate(text.charAt(offset))) {
            return 2;
        }
        return 1;
    }

    private static boolean isNewlineOnly(String word) {
        for (int i = 0; i < word.length(); i++) {
            if (word.charAt(i) != '\n') {
                return false;
            }
        }
        return !word.isEmpty();
    }

    private static Map<String, Integer> buildMergeRanks(String[] merges) {
        Map<String, Integer> ranks = new HashMap<>();
        if (merges == null) {
            return ranks;
        }
        for (int i = 0; i < merges.length; i++) {
            int space = merges[i].indexOf(' ');
            if (space > 0) {
                String key = merges[i].substring(0, space) + merges[i].substring(space + 1);
                ranks.put(key, i);
            }
        }
        return ranks;
    }

    private static final class Symbol {
        String text;
        boolean empty;
        int prev;
        int next;

        Symbol(String text, int prev, int next) {
            this.text = text;
            this.prev = prev;
            this.next = next;
        }
    }

    private static final class Bigram {
        final int left;
        final int right;
        final String leftText;
        final String rightText;
        final int rank;

        Bigram(int left, int right, String leftText, String rightText, int rank) {
            this.left = left;
            this.right = right;
            this.leftText = leftText;
            this.rightText = rightText;
            this.rank = rank;
        }
    }

    private static final class PriorityBigramQueue {
        private final List<Bigram> items = new ArrayList<>();

        boolean isEmpty() {
            return items.isEmpty();
        }

        void push(Bigram bigram) {
            items.add(bigram);
        }

        Bigram pop() {
            int best = 0;
            for (int i = 1; i < items.size(); i++) {
                if (items.get(i).rank < items.get(best).rank) {
                    best = i;
                }
            }
            return items.remove(best);
        }
    }
}
