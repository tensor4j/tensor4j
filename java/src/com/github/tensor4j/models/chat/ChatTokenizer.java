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

import com.github.tensor4j.runtime.gguf.GgufArrayValue;
import com.github.tensor4j.runtime.gguf.GgufHeader;
import com.github.tensor4j.runtime.gguf.GgufKvEntry;
import com.github.tensor4j.runtime.gguf.GgufType;

/** Text ↔ token ids from GGUF {@code tokenizer.ggml.*} metadata (llama.cpp llama_vocab). */
public final class ChatTokenizer {

    private final String[] tokens;
    private final int bosId;
    private final int eosId;
    private final BpePreType preType;
    private final BpeSplitMode splitMode;
    private final BpeTokenizer bpe;

    private ChatTokenizer(
            String[] tokens,
            int bosId,
            int eosId,
            BpePreType preType,
            BpeSplitMode splitMode,
            BpeTokenizer bpe) {
        this.tokens = tokens.clone();
        this.bosId = bosId;
        this.eosId = eosId;
        this.preType = preType;
        this.splitMode = splitMode;
        this.bpe = bpe;
    }

    public static ChatTokenizer fromGguf(GgufHeader header) {
        return fromGguf(header, ChatTokenizerOptions.defaults());
    }

    public static ChatTokenizer fromGguf(GgufHeader header, ChatTokenizerOptions options) {
        String model = stringKv(header, "tokenizer.ggml.model", "llama");
        String[] vocabTokens = stringArrayKv(header, "tokenizer.ggml.tokens");
        if (vocabTokens.length == 0) {
            throw new IllegalArgumentException("missing tokenizer.ggml.tokens");
        }
        int bos = intKv(header, "tokenizer.ggml.bos_token_id", 0);
        int eos = intKv(header, "tokenizer.ggml.eos_token_id", vocabTokens.length - 1);
        String pre = stringKv(header, "tokenizer.ggml.pre", "default");
        BpePreType preType = BpePreType.fromPre(pre);
        boolean ignoreMerges = boolKv(header, "tokenizer.ggml.ignore_merges", preType.defaultIgnoreMerges());
        BpeTokenizer bpe = null;
        if ("gpt2".equals(model) || "llama".equals(model)) {
            String[] merges = stringArrayKv(header, "tokenizer.ggml.merges");
            bpe = new BpeTokenizer(vocabTokens, merges, preType, options.splitMode(), ignoreMerges);
        }
        return new ChatTokenizer(vocabTokens, bos, eos, preType, options.splitMode(), bpe);
    }

    public BpePreType preType() {
        return preType;
    }

    public BpeSplitMode splitMode() {
        return splitMode;
    }

    public int bosId() {
        return bosId;
    }

    public int eosId() {
        return eosId;
    }

    public int vocabSize() {
        return tokens.length;
    }

    public String tokenText(int id) {
        if (id < 0 || id >= tokens.length) {
            throw new IllegalArgumentException("token id out of range " + id);
        }
        return tokens[id];
    }

    /** Encode text to token ids (no automatic BOS/EOS). */
    public int[] encode(String text) {
        if (bpe == null) {
            return encodeLiteral(text);
        }
        return bpe.encode(text);
    }

    /** Encode with leading BOS when configured. */
    public int[] encodeWithBos(String text) {
        int[] body = encode(text);
        if (bosId < 0) {
            return body;
        }
        int[] out = new int[body.length + 1];
        out[0] = bosId;
        System.arraycopy(body, 0, out, 1, body.length);
        return out;
    }

    public String decode(int[] ids) {
        if (bpe == null) {
            return decodeLiteral(ids);
        }
        return bpe.decode(ids);
    }

    private int[] encodeLiteral(String text) {
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].equals(text)) {
                return new int[] {i};
            }
        }
        int[] out = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            String ch = text.substring(i, i + 1);
            out[i] = tokenId(ch);
        }
        return out;
    }

    private String decodeLiteral(int[] ids) {
        StringBuilder out = new StringBuilder();
        for (int id : ids) {
            out.append(tokenText(id));
        }
        return out.toString();
    }

    private int tokenId(String text) {
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].equals(text)) {
                return i;
            }
        }
        throw new IllegalArgumentException("unknown token text " + text);
    }

    private static String stringKv(GgufHeader header, String key, String defaultValue) {
        GgufKvEntry entry = header.findKv(key);
        if (entry == null) {
            return defaultValue;
        }
        return (String) entry.value();
    }

    private static int intKv(GgufHeader header, String key, int defaultValue) {
        GgufKvEntry entry = header.findKv(key);
        if (entry == null) {
            return defaultValue;
        }
        return ((Number) entry.value()).intValue();
    }

    private static boolean boolKv(GgufHeader header, String key, boolean defaultValue) {
        GgufKvEntry entry = header.findKv(key);
        if (entry == null) {
            return defaultValue;
        }
        return (Boolean) entry.value();
    }

    private static String[] stringArrayKv(GgufHeader header, String key) {
        GgufKvEntry entry = header.findKv(key);
        if (entry == null) {
            return new String[0];
        }
        GgufArrayValue array = (GgufArrayValue) entry.value();
        if (array.elementType() != GgufType.STRING) {
            throw new IllegalArgumentException(key + " must be string array");
        }
        String[] out = new String[array.elements().length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (String) array.elements()[i];
        }
        return out;
    }
}
