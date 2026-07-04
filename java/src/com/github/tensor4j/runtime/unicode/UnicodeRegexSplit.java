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

/**
 * llama.cpp {@code unicode_regex_split} — codepoint splitters with collapsed-regex fallback.
 */
public final class UnicodeRegexSplit {

    private UnicodeRegexSplit() {
    }

    public static List<String> split(String text, String[] regexExprs) {
        if (regexExprs.length == 0) {
            if (text.isEmpty()) {
                return List.of();
            }
            return List.of(text);
        }
        int[] cpts = UnicodeCpt.codepointsFromUtf8(text);
        if (cpts.length == 0) {
            return List.of();
        }
        String collapsed = UnicodeCollapse.needsCollapse(regexExprs) ? UnicodeCollapse.collapseText(cpts) : null;
        String wide = collapsed != null ? collapsed : codepointsToChars(cpts);
        List<Integer> offsets = new ArrayList<>();
        offsets.add(cpts.length);
        for (String regexExpr : regexExprs) {
            List<Integer> custom = UnicodeRegexSplitCustom.split(text, regexExpr, offsets);
            if (!custom.isEmpty()) {
                offsets = custom;
                continue;
            }
            offsets = splitFallback(wide, regexExpr, offsets);
        }
        return offsetsToWords(cpts, offsets);
    }

    private static List<Integer> splitFallback(String wide, String regexExpr, List<Integer> offsets) {
        if (regexUsesUnicodeCategory(regexExpr)) {
            String regexCollapsed = UnicodeCollapse.collapseRegex(regexExpr);
            return UnicodeRegexSplitStl.split(wide, regexCollapsed, offsets);
        }
        return UnicodeRegexSplitStl.split(wide, regexExpr, offsets);
    }

    private static String codepointsToChars(int[] cpts) {
        char[] chars = new char[cpts.length];
        for (int i = 0; i < cpts.length; i++) {
            int cpt = cpts[i];
            if (cpt > 0x7F && UnicodeCptFlags.fromCodepoint(cpt).isWhitespace()) {
                chars[i] = 0x0B;
            } else if (cpt <= 0xFFFF) {
                chars[i] = (char) cpt;
            } else {
                chars[i] = (char) 0xD0;
            }
        }
        return new String(chars);
    }

    private static boolean regexUsesUnicodeCategory(String regexExpr) {
        return regexExpr.contains("\\p{");
    }

    private static List<String> offsetsToWords(int[] cpts, List<Integer> offsets) {
        List<String> words = new ArrayList<>(offsets.size());
        int start = 0;
        for (int offset : offsets) {
            if (offset > 0) {
                words.add(UnicodeCpt.sliceToUtf8(cpts, start, start + offset));
            }
            start += offset;
        }
        return words;
    }
}
