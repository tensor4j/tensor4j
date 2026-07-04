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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Collapsed byte view for regex ({@code unicode_regex_split} collapse path in llama.cpp). */
final class UnicodeCollapse {

    private static final int CPT_NUMBER = 0xD1;
    private static final int CPT_LETTER = 0xD2;
    private static final int CPT_PUNCTUATION = 0xD3;
    private static final int CPT_ACCENT = 0xD4;
    private static final int CPT_SYMBOL = 0xD5;
    private static final int CPT_FALLBACK = 0xD0;
    private static final int CPT_WHITESPACE = 0x0B;

    private static final Map<String, Integer> CATEGORY_TOKEN = Map.of(
            "\\p{N}", UnicodeCategory.NUMBER,
            "\\p{L}", UnicodeCategory.LETTER,
            "\\p{P}", UnicodeCategory.PUNCTUATION,
            "\\p{M}", UnicodeCategory.ACCENT,
            "\\p{S}", UnicodeCategory.SYMBOL,
            "\\p{Lu}", UnicodeCategory.LETTER,
            "\\p{Ll}", UnicodeCategory.LETTER,
            "\\p{Lt}", UnicodeCategory.LETTER,
            "\\p{Lm}", UnicodeCategory.LETTER,
            "\\p{Lo}", UnicodeCategory.LETTER);

    private UnicodeCollapse() {
    }

    static boolean needsCollapse(String[] regexExprs) {
        for (String regex : regexExprs) {
            for (String token : CATEGORY_TOKEN.keySet()) {
                if (regex.contains(token)) {
                    return true;
                }
            }
        }
        return false;
    }

    static String collapseText(int[] cpts) {
        char[] out = new char[cpts.length];
        for (int i = 0; i < cpts.length; i++) {
            out[i] = (char) collapseCodepoint(cpts[i]);
        }
        return new String(out);
    }

    static String collapseRegex(String regexExpr) {
        StringBuilder out = new StringBuilder();
        boolean inside = false;
        for (int i = 0; i < regexExpr.length(); i++) {
            char ch = regexExpr.charAt(i);
            if (ch == '[' && (i == 0 || regexExpr.charAt(i - 1) != '\\')) {
                out.append('[');
                inside = true;
                continue;
            }
            if (inside && ch == ']' && regexExpr.charAt(i - 1) != '\\') {
                out.append(']');
                inside = false;
                continue;
            }
            if (regexExpr.startsWith("\\p{", i)) {
                int close = regexExpr.indexOf('}', i + 3);
                if (close > i && close <= i + 10) {
                    String pat = regexExpr.substring(i, close + 1);
                    Integer cat = CATEGORY_TOKEN.get(pat);
                    if (cat != null) {
                        if (!inside) {
                            out.append('[');
                        }
                        out.append((char) categoryByte(cat));
                        out.append(categoryAsciiRange(cat));
                        if (!inside) {
                            out.append(']');
                        }
                        i = close;
                        continue;
                    }
                }
            }
            out.append(ch);
        }
        return out.toString();
    }

    private static int collapseCodepoint(int cpt) {
        if (cpt < 128) {
            return cpt;
        }
        UnicodeCptFlags flags = UnicodeCptFlags.fromCodepoint(cpt);
        if (flags.isWhitespace()) {
            return CPT_WHITESPACE;
        }
        if (flags.isNumber()) {
            return CPT_NUMBER;
        }
        if (flags.isLetter()) {
            return CPT_LETTER;
        }
        if (flags.isPunctuation()) {
            return CPT_PUNCTUATION;
        }
        if (flags.isAccentMark()) {
            return CPT_ACCENT;
        }
        if (flags.isSymbol()) {
            return CPT_SYMBOL;
        }
        return CPT_FALLBACK;
    }

    private static int categoryByte(int category) {
        if (category == UnicodeCategory.NUMBER) {
            return CPT_NUMBER;
        }
        if (category == UnicodeCategory.LETTER) {
            return CPT_LETTER;
        }
        if (category == UnicodeCategory.PUNCTUATION) {
            return CPT_PUNCTUATION;
        }
        if (category == UnicodeCategory.ACCENT) {
            return CPT_ACCENT;
        }
        return CPT_SYMBOL;
    }

    private static String categoryAsciiRange(int category) {
        if (category == UnicodeCategory.NUMBER) {
            return "0-9";
        }
        if (category == UnicodeCategory.LETTER) {
            return "A-Za-z";
        }
        if (category == UnicodeCategory.PUNCTUATION) {
            return "!-#%-*,-/:-;?-@\\[\\-\\\\\\]_\\{\\}";
        }
        if (category == UnicodeCategory.ACCENT) {
            return "";
        }
        return "$+<=>^`|";
    }

    private static final class UnicodeCategory {
        private static final int NUMBER = 1;
        private static final int LETTER = 2;
        private static final int PUNCTUATION = 3;
        private static final int ACCENT = 4;
        private static final int SYMBOL = 5;

        private UnicodeCategory() {
        }
    }
}
