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
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Regex pre-split for BPE ({@code unicode_regex_split} in llama.cpp unicode.cpp). */
final class BpeRegexSplit {

    private static final int PATTERN_FLAGS = Pattern.UNICODE_CHARACTER_CLASS;

    private BpeRegexSplit() {
    }

    static List<String> split(String text, BpePreType preType) {
        if (preType == BpePreType.LLAMA_SPM || preType.regexes().length == 0) {
            if (text.isEmpty()) {
                return List.of();
            }
            return List.of(text);
        }
        List<String> parts = new ArrayList<>();
        parts.add(text);
        for (String regex : preType.regexes()) {
            Pattern pattern = Pattern.compile(regex, PATTERN_FLAGS);
            List<String> next = new ArrayList<>();
            for (String part : parts) {
                next.addAll(splitOnce(part, pattern));
            }
            parts = next;
        }
        return parts;
    }

    private static List<String> splitOnce(String text, Pattern pattern) {
        List<String> out = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        int pos = 0;
        while (matcher.find()) {
            if (matcher.start() > pos) {
                out.add(text.substring(pos, matcher.start()));
            }
            out.add(matcher.group());
            pos = matcher.end();
        }
        if (pos < text.length()) {
            out.add(text.substring(pos));
        }
        if (out.isEmpty() && !text.isEmpty()) {
            out.add(text);
        }
        return out;
    }
}
