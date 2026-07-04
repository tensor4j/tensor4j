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

import com.github.tensor4j.runtime.unicode.UnicodeRegexPatterns;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * BPE pre-tokenizer profile ({@code llama_vocab_pre_type} + {@code tokenizer.ggml.pre} in llama-vocab.cpp).
 */
public enum BpePreType {

    DEFAULT(
            new String[] {
                "[\\p{P}\\$\\+<=>\\^~\\|]+",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                "\\p{N}+",
                "[0-9][0-9][0-9]",
            },
            true,
            false),
    GPT2(
            new String[] {
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            },
            true,
            false),
    LLAMA3(
            new String[] {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                        + "\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            true),
    JAIS2(
            new String[] {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                        + "\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)"
                        + "|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)"
                        + "|\\s{1,2}(?!\\S)|\\s{1}",
            },
            true,
            false),
    QWEN2(
            new String[] {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                        + "\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            false),
    QWEN35(
            new String[] {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                        + "[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            false),
    GEMMA4(
            new String[] {
                "[^\\n]+|[\\n]+",
            },
            false,
            false),
    FALCON(
            new String[] {
                "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                "[0-9][0-9][0-9]",
            },
            true,
            false),
    STARCODER(
            new String[] {
                "\\p{N}",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            },
            true,
            false),
    LLAMA_SPM(
            new String[0],
            false,
            false),
    KIMI_K2(
            new String[] {UnicodeRegexPatterns.KIMI_K2},
            true,
            false),
    AFMOE(
            new String[] {
                UnicodeRegexPatterns.AFMOE_DIGITS,
                "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
                "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            false),
    TINY_AYA(
            new String[] {
                UnicodeRegexPatterns.TINY_AYA_DIGITS,
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            false),
    BAILINGMOE(
            new String[] {UnicodeRegexPatterns.BAILINGMOE},
            true,
            false),
    SEED_CODER(
            new String[] {UnicodeRegexPatterns.SEED_CODER},
            true,
            false),
    GROK2(
            new String[] {UnicodeRegexPatterns.GROK2},
            true,
            false),
    EXAONE_MOE(
            new String[] {UnicodeRegexPatterns.EXAONE_MOE},
            true,
            false),
    CHATGLM4(
            new String[] {UnicodeRegexPatterns.LLAMA3},
            true,
            false),
    GPT4O(
            new String[] {
                "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            false),
    TEKKEN(
            new String[] {
                "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            true),
    CHAMELEON(
            new String[] {
                "<sentinel:[0-9]+>",
                "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
                "([\\t\\n]| | )",
                "\\p{N}",
                "[\\p{P}!-/:-@\\[-`{-~]",
                UnicodeRegexPatterns.GPT2,
            },
            true,
            false),
    SUPERBPE(
            new String[] {
                "\\p{N}+",
                "(?=(\\d{3})+(?!\\d))",
            },
            true,
            false),
    PORO(
            new String[] {
                " ?[^(\\s|.,!?…。，、।۔،)]+",
            },
            true,
            false),
    BLOOM(
            new String[] {
                " ?[^(\\s|.,!?…。，、।۔،)]+",
            },
            true,
            false),
    MINICPM5(
            new String[] {
                "\\p{N}{1,3}",
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            true),
    SARVAM_MOE(
            new String[] {UnicodeRegexPatterns.GEMMA4_NEWLINES},
            false,
            false),
    DEEPSEEK_LLM(
            new String[] {
                "[\r\n]",
                "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                "\\s+$",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}+",
            },
            true,
            false),
    DEEPSEEK_CODER(
            new String[] {
                "[\r\n]",
                "\\s?\\p{L}+",
                "\\s?\\p{P}+",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}",
            },
            true,
            false),
    DEEPSEEK3(
            new String[] {
                "\\p{N}{1,3}",
                "[一-龥぀-ゟ゠-ヿ]+",
                "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            false),
    YOUTU(
            new String[] {
                "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            },
            true,
            true),
    VIKING(
            new String[] {
                " ?[^(\\s|.,!?…。，、।۔،)]+",
                "\\p{N}",
            },
            true,
            false),
    WHITESPACE(
            new String[] {
                "\\S+",
            },
            false,
            false);

    private static final Map<String, BpePreType> BY_PRE = buildPreMap();

    private final String[] regexes;
    private final boolean byteEncode;
    private final boolean defaultIgnoreMerges;

    BpePreType(String[] regexes, boolean byteEncode, boolean defaultIgnoreMerges) {
        this.regexes = regexes.clone();
        this.byteEncode = byteEncode;
        this.defaultIgnoreMerges = defaultIgnoreMerges;
    }

    public String[] regexes() {
        return regexes.clone();
    }

    public boolean byteEncode() {
        return byteEncode;
    }

    public boolean defaultIgnoreMerges() {
        return defaultIgnoreMerges;
    }

    public static BpePreType fromPre(String pre) {
        if (pre == null || pre.isEmpty() || "default".equals(pre)) {
            return DEFAULT;
        }
        BpePreType mapped = BY_PRE.get(pre.toLowerCase(Locale.ROOT));
        if (mapped != null) {
            return mapped;
        }
        throw new IllegalArgumentException("unknown pre-tokenizer type: " + pre);
    }

    private static Map<String, BpePreType> buildPreMap() {
        Map<String, BpePreType> map = new HashMap<>();
        map.put("llama-spm", LLAMA_SPM);
        map.put("gpt-2", GPT2);
        map.put("mpt", GPT2);
        map.put("olmo", GPT2);
        map.put("jais", GPT2);
        map.put("trillion", GPT2);
        map.put("granite-docling", GPT2);
        map.put("llama3", LLAMA3);
        map.put("llama-v3", LLAMA3);
        map.put("llama-bpe", LLAMA3);
        map.put("falcon3", LLAMA3);
        map.put("falcon-h1", LLAMA3);
        map.put("pixtral", LLAMA3);
        map.put("midm-2.0", LLAMA3);
        map.put("lfm2", LLAMA3);
        map.put("jina-v5-nano", LLAMA3);
        map.put("dbrx", LLAMA3);
        map.put("smaug-bpe", LLAMA3);
        map.put("jais-2", JAIS2);
        map.put("qwen2", QWEN2);
        map.put("deepseek-r1-qwen", QWEN2);
        map.put("kormo", QWEN2);
        map.put("f2llmv2", QWEN2);
        map.put("megrez", QWEN2);
        map.put("stablelm2", QWEN2);
        map.put("hunyuan", QWEN2);
        map.put("solar-open", QWEN2);
        map.put("qwen35", QWEN35);
        map.put("gemma4", GEMMA4);
        map.put("falcon", FALCON);
        map.put("starcoder", STARCODER);
        map.put("refact", STARCODER);
        map.put("command-r", STARCODER);
        map.put("smollm", STARCODER);
        map.put("codeshell", STARCODER);
        map.put("exaone", STARCODER);
        map.put("minerva-7b", STARCODER);
        map.put("mellum2", STARCODER);
        map.put("mellum", GPT2);
        map.put("deepseek-llm", DEEPSEEK_LLM);
        map.put("deepseek-coder", DEEPSEEK_CODER);
        map.put("deepseek-v3", DEEPSEEK3);
        map.put("hunyuan-dense", DEEPSEEK3);
        map.put("joyai-llm", DEEPSEEK3);
        map.put("youtu", YOUTU);
        map.put("kimi-k2", KIMI_K2);
        map.put("afmoe", AFMOE);
        map.put("tiny_aya", TINY_AYA);
        map.put("cohere2moe", TINY_AYA);
        map.put("bailingmoe", BAILINGMOE);
        map.put("bailingmoe2", BAILINGMOE);
        map.put("llada-moe", BAILINGMOE);
        map.put("seed-coder", SEED_CODER);
        map.put("grok-2", GROK2);
        map.put("exaone-moe", EXAONE_MOE);
        map.put("exaone4", GPT2);
        map.put("chatglm-bpe", CHATGLM4);
        map.put("glm4", CHATGLM4);
        map.put("gpt-4o", GPT4O);
        map.put("llama4", GPT4O);
        map.put("kanana2", GPT4O);
        map.put("talkie", GPT4O);
        map.put("minimax-m2", GPT4O);
        map.put("granite-embed-multi-97m", GPT4O);
        map.put("tekken", TEKKEN);
        map.put("chameleon", CHAMELEON);
        map.put("superbpe", SUPERBPE);
        map.put("poro-chat", PORO);
        map.put("bloom", BLOOM);
        map.put("gpt3-finnish", BLOOM);
        map.put("minicpm5", MINICPM5);
        map.put("sarvam-moe", SARVAM_MOE);
        map.put("granite-embed-multi-311m", GEMMA4);
        map.put("viking", VIKING);
        map.put("jina-es", GPT2);
        map.put("jina-de", GPT2);
        map.put("jina-v2-es", GPT2);
        map.put("jina-v2-de", GPT2);
        map.put("gigachat", GPT2);
        map.put("a.x-4.0", GPT2);
        map.put("modern-bert", GPT2);
        map.put("phi-2", GPT2);
        map.put("whitespace", WHITESPACE);
        map.put("jina-v1-en", GPT2);
        map.put("jina-v2-code", GPT2);
        map.put("roberta-bpe", GPT2);
        return map;
    }
}
