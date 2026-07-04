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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Collapsed-regex fallback ({@code unicode_regex_split_stl} in llama.cpp). */
final class UnicodeRegexSplitStl {

    private UnicodeRegexSplitStl() {
    }

    static List<Integer> split(String text, String regex, List<Integer> offsets) {
        Pattern pattern = Pattern.compile(regex);
        List<Integer> out = new ArrayList<>();
        int start = 0;
        for (int offset : offsets) {
            splitSegment(text, start, offset, pattern, out);
            start += offset;
        }
        return out;
    }

    private static void splitSegment(String text, int start, int length, Pattern pattern, List<Integer> out) {
        int segmentStart = start;
        int segmentEnd = start + length;
        Matcher matcher = pattern.matcher(text);
        matcher.region(segmentStart, segmentEnd);
        int localStart = 0;
        while (matcher.find()) {
            if (matcher.start() > localStart) {
                out.add(matcher.start() - localStart);
            }
            out.add(matcher.end() - matcher.start());
            localStart = matcher.end();
        }
        if (localStart < length) {
            out.add(length - localStart);
        }
    }
}
