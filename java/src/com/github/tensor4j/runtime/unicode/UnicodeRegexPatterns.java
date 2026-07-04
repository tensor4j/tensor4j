/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.unicode;

/** Known BPE regex strings from llama-vocab.cpp / unicode.cpp. */
public final class UnicodeRegexPatterns {

    public static final String GPT2 =
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";

    public static final String LLAMA3 =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                    + "\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    public static final String QWEN2 =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                    + "\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    public static final String QWEN35 =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                    + "[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    public static final String GEMMA4_NEWLINES = "[^\\n]+|[\\n]+";

    /** llama3 legacy case-insensitive variant (still uses llama3 custom splitter). */
    public static final String LLAMA3_LEGACY =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    /** Kimi-K2 trigger — full K2 logic runs ({@code unicode_regex_split_custom_kimi_k2}). */
    public static final String KIMI_K2 = "\\p{Han}+";

    public static final String AFMOE_DIGITS = "\\p{AFMoE_digits}";

    public static final String TINY_AYA_DIGITS = "\\d{1,3}(?=(?:\\d{3})*\\b)";

    public static final String GROK2 = QWEN2;

    public static final String SEED_CODER =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                    + "\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    public static final String BAILINGMOE =
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
                    + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";

    public static final String EXAONE_MOE =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?"
                    + "(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";

    private UnicodeRegexPatterns() {
    }

    public static boolean isLlama3Family(String regexExpr) {
        return LLAMA3.equals(regexExpr) || LLAMA3_LEGACY.equals(regexExpr);
    }

    public static boolean isQwen2Family(String regexExpr) {
        return QWEN2.equals(regexExpr) || GROK2.equals(regexExpr);
    }
}
