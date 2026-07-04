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

/** Custom codepoint splitters ({@code unicode_regex_split_custom} in llama.cpp). */
final class UnicodeRegexSplitCustom {

    private static final int OUT_OF_RANGE = -1;

    private UnicodeRegexSplitCustom() {
    }

    static List<Integer> split(String text, String regexExpr, List<Integer> offsets) {
        if (UnicodeRegexPatterns.GPT2.equals(regexExpr)) {
            return splitGpt2(text, offsets);
        }
        if (UnicodeRegexPatterns.isLlama3Family(regexExpr)) {
            return splitLlama3(text, offsets);
        }
        if (UnicodeRegexPatterns.isQwen2Family(regexExpr)
                || UnicodeRegexPatterns.SEED_CODER.equals(regexExpr)) {
            return splitQwen2(text, offsets);
        }
        if (UnicodeRegexPatterns.QWEN35.equals(regexExpr)) {
            return splitQwen35(text, offsets);
        }
        if (UnicodeRegexPatterns.GEMMA4_NEWLINES.equals(regexExpr)) {
            return splitNewlines(text, offsets);
        }
        if (UnicodeRegexPatterns.KIMI_K2.equals(regexExpr)) {
            return splitKimiK2(text, offsets);
        }
        if (UnicodeRegexPatterns.AFMOE_DIGITS.equals(regexExpr)
                || UnicodeRegexPatterns.TINY_AYA_DIGITS.equals(regexExpr)) {
            return splitAfmoe(text, offsets);
        }
        return List.of();
    }

    private static List<Integer> splitGpt2(String text, List<Integer> offsets) {
        return splitWithRunner(text, offsets, new Gpt2Runner());
    }

    private static List<Integer> splitLlama3(String text, List<Integer> offsets) {
        return splitWithRunner(text, offsets, new Llama3Runner());
    }

    private static List<Integer> splitQwen2(String text, List<Integer> offsets) {
        return splitWithRunner(text, offsets, new Qwen2Runner());
    }

    private static List<Integer> splitQwen35(String text, List<Integer> offsets) {
        return splitWithRunner(text, offsets, new Qwen35Runner());
    }

    private static List<Integer> splitNewlines(String text, List<Integer> offsets) {
        int[] cpts = UnicodeCpt.codepointsFromUtf8(text);
        List<Integer> out = new ArrayList<>();
        int start = 0;
        for (int offset : offsets) {
            int ini = start;
            int end = start + offset;
            int pos = ini;
            while (pos < end) {
                boolean isNewline = cpts[pos] == '\n';
                int runStart = pos;
                while (pos < end && (cpts[pos] == '\n') == isNewline) {
                    pos++;
                }
                out.add(pos - runStart);
            }
            start = end;
        }
        return out;
    }

    private static List<Integer> splitKimiK2(String text, List<Integer> offsets) {
        return splitWithRunner(text, offsets, new KimiK2Runner());
    }

    /** AFMoE / tiny_aya digit grouping ({@code unicode_regex_split_custom_afmoe}). */
    private static List<Integer> splitAfmoe(String text, List<Integer> offsets) {
        int[] cpts = UnicodeCpt.codepointsFromUtf8(text);
        List<Integer> out = new ArrayList<>();
        int start = 0;
        for (int offset : offsets) {
            int ini = start;
            int end = start + offset;
            int prevEnd = ini;
            for (int pos = ini; pos < end; ) {
                UnicodeCptFlags flags = flagsAt(cpts, ini, end, pos);
                if (flags.isNumber()) {
                    int digitStart = pos;
                    int digitCount = 0;
                    while (flagsAt(cpts, ini, end, pos).isNumber()) {
                        digitCount++;
                        pos++;
                    }
                    int remainder = digitCount % 3;
                    int current = digitStart;
                    if (remainder > 0) {
                        prevEnd = addToken(out, prevEnd, current + remainder);
                        current += remainder;
                    }
                    while (current < digitStart + digitCount) {
                        prevEnd = addToken(out, prevEnd, current + 3);
                        current += 3;
                    }
                } else {
                    pos++;
                }
            }
            if (prevEnd < end) {
                addToken(out, prevEnd, end);
            }
            start = end;
        }
        return out;
    }

    private static UnicodeCptFlags flagsAt(int[] cpts, int ini, int end, int pos) {
        if (ini <= pos && pos < end) {
            return UnicodeCptFlags.fromCodepoint(cpts[pos]);
        }
        return UnicodeCptFlags.UNDEFINED;
    }

    private static int addToken(List<Integer> out, int prevEnd, int end) {
        if (prevEnd < end) {
            out.add(end - prevEnd);
        }
        return end;
    }

    private static List<Integer> splitWithRunner(String text, List<Integer> offsets, Runner runner) {
        int[] cpts = UnicodeCpt.codepointsFromUtf8(text);
        List<Integer> out = new ArrayList<>();
        int start = 0;
        for (int offset : offsets) {
            Segment segment = new Segment(cpts, start, start + offset, out);
            runner.run(segment);
            start += offset;
        }
        return out;
    }

    private interface Runner {
        void run(Segment segment);
    }

    private static final class Segment {
        private final int[] cpts;
        private final int ini;
        private final int end;
        private final List<Integer> out;
        private int prevEnd;

        private Segment(int[] cpts, int ini, int end, List<Integer> out) {
            this.cpts = cpts;
            this.ini = ini;
            this.end = end;
            this.out = out;
            this.prevEnd = ini;
        }

        private int cpt(int pos) {
            if (ini <= pos && pos < end) {
                return cpts[pos];
            }
            return OUT_OF_RANGE;
        }

        private UnicodeCptFlags flags(int pos) {
            if (ini <= pos && pos < end) {
                return UnicodeCptFlags.fromCodepoint(cpts[pos]);
            }
            return UnicodeCptFlags.UNDEFINED;
        }

        private void addToken(int endPos) {
            if (prevEnd < endPos) {
                out.add(endPos - prevEnd);
            }
            prevEnd = endPos;
        }

        private void run(RunnerBody body) {
            for (int pos = ini; pos < end; ) {
                pos = body.step(this, pos);
            }
        }
    }

    private interface RunnerBody {
        int step(Segment s, int pos);
    }

    private static final class Gpt2Runner implements Runner {
        public void run(Segment s) {
            s.run(Gpt2Runner::step);
        }

        private static int step(Segment s, int pos) {
            int cpt = s.cpt(pos);
            UnicodeCptFlags flags = s.flags(pos);
            if (cpt == '\'' && s.cpt(pos + 1) != OUT_OF_RANGE) {
                int next = s.cpt(pos + 1);
                if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                    s.addToken(pos + 2);
                    return pos + 2;
                }
                if (s.cpt(pos + 2) != OUT_OF_RANGE) {
                    int n2 = s.cpt(pos + 2);
                    if ((next == 'r' && n2 == 'e') || (next == 'v' && n2 == 'e') || (next == 'l' && n2 == 'l')) {
                        s.addToken(pos + 3);
                        return pos + 3;
                    }
                }
            }
            UnicodeCptFlags flags2 = cpt == ' ' ? s.flags(pos + 1) : flags;
            if (flags2.isLetter()) {
                pos += cpt == ' ' ? 1 : 0;
                while (s.flags(pos).isLetter()) {
                    pos++;
                }
                s.addToken(pos);
                return pos;
            }
            if (flags2.isNumber()) {
                pos += cpt == ' ' ? 1 : 0;
                while (s.flags(pos).isNumber()) {
                    pos++;
                }
                s.addToken(pos);
                return pos;
            }
            if (!flags2.isWhitespace() && !flags2.isLetter() && !flags2.isNumber() && flags2.hasCategory()) {
                pos += cpt == ' ' ? 1 : 0;
                UnicodeCptFlags f = s.flags(pos);
                while (!f.isWhitespace() && !f.isLetter() && !f.isNumber() && f.hasCategory()) {
                    f = s.flags(++pos);
                }
                s.addToken(pos);
                return pos;
            }
            int ws = countWhitespace(s, pos);
            if (ws > 1 && s.cpt(pos + ws) != OUT_OF_RANGE) {
                s.addToken(pos + ws - 1);
                return pos + ws - 1;
            }
            if (ws > 0) {
                s.addToken(pos + ws);
                return pos + ws;
            }
            s.addToken(pos + 1);
            return pos + 1;
        }
    }

    private static final class Llama3Runner implements Runner {
        public void run(Segment s) {
            s.run(Llama3Runner::step);
        }

        private static int step(Segment s, int pos) {
            int cpt = s.cpt(pos);
            UnicodeCptFlags flags = s.flags(pos);
            if (cpt == '\'' && s.cpt(pos + 1) != OUT_OF_RANGE) {
                int next = UnicodeCpt.toLower(s.cpt(pos + 1));
                if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                    s.addToken(pos + 2);
                    return pos + 2;
                }
                if (s.cpt(pos + 2) != OUT_OF_RANGE) {
                    int n2 = UnicodeCpt.toLower(s.cpt(pos + 2));
                    if ((next == 'r' && n2 == 'e') || (next == 'v' && n2 == 'e') || (next == 'l' && n2 == 'l')) {
                        s.addToken(pos + 3);
                        return pos + 3;
                    }
                }
            }
            if (cpt != '\r' && cpt != '\n' && !flags.isNumber()) {
                if (flags.isLetter() || s.flags(pos + 1).isLetter()) {
                    pos++;
                    while (s.flags(pos).isLetter()) {
                        pos++;
                    }
                    s.addToken(pos);
                    return pos;
                }
            }
            if (flags.isNumber()) {
                int ini = pos;
                while (s.flags(pos).isNumber()) {
                    if (++pos - ini >= 3) {
                        s.addToken(pos);
                        ini = pos;
                    }
                }
                s.addToken(pos);
                return pos;
            }
            return whitespaceAndSymbolTail(s, pos, cpt, flags);
        }
    }

    private static final class Qwen2Runner implements Runner {
        public void run(Segment s) {
            s.run(Qwen2Runner::step);
        }

        private static int step(Segment s, int pos) {
            int cpt = s.cpt(pos);
            UnicodeCptFlags flags = s.flags(pos);
            if (cpt == '\'' && s.cpt(pos + 1) != OUT_OF_RANGE) {
                int next = UnicodeCpt.toLower(s.cpt(pos + 1));
                if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                    s.addToken(pos + 2);
                    return pos + 2;
                }
                if (s.cpt(pos + 2) != OUT_OF_RANGE) {
                    int n2 = UnicodeCpt.toLower(s.cpt(pos + 2));
                    if ((next == 'r' && n2 == 'e') || (next == 'v' && n2 == 'e') || (next == 'l' && n2 == 'l')) {
                        s.addToken(pos + 3);
                        return pos + 3;
                    }
                }
            }
            if (cpt != '\r' && cpt != '\n' && !flags.isNumber()) {
                if (flags.isLetter() || s.flags(pos + 1).isLetter()) {
                    pos++;
                    while (s.flags(pos).isLetter()) {
                        pos++;
                    }
                    s.addToken(pos);
                    return pos;
                }
            }
            if (flags.isNumber()) {
                s.addToken(pos + 1);
                return pos + 1;
            }
            return whitespaceAndSymbolTail(s, pos, cpt, flags);
        }
    }

    private static final class Qwen35Runner implements Runner {
        public void run(Segment s) {
            s.run(Qwen35Runner::step);
        }

        private static int step(Segment s, int pos) {
            int cpt = s.cpt(pos);
            UnicodeCptFlags flags = s.flags(pos);
            if (cpt == '\'' && s.cpt(pos + 1) != OUT_OF_RANGE) {
                int next = UnicodeCpt.toLower(s.cpt(pos + 1));
                if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                    s.addToken(pos + 2);
                    return pos + 2;
                }
                if (s.cpt(pos + 2) != OUT_OF_RANGE) {
                    int n2 = UnicodeCpt.toLower(s.cpt(pos + 2));
                    if ((next == 'r' && n2 == 'e') || (next == 'v' && n2 == 'e') || (next == 'l' && n2 == 'l')) {
                        s.addToken(pos + 3);
                        return pos + 3;
                    }
                }
            }
            if (cpt != '\r' && cpt != '\n' && !flags.isNumber()) {
                if (flags.isLetter() || flags.isAccentMark() || s.flags(pos + 1).isAccentMark()
                        || s.flags(pos + 1).isLetter()) {
                    pos++;
                    while (s.flags(pos).isLetter() || s.flags(pos).isAccentMark()) {
                        pos++;
                    }
                    s.addToken(pos);
                    return pos;
                }
            }
            if (flags.isNumber()) {
                s.addToken(pos + 1);
                return pos + 1;
            }
            UnicodeCptFlags flags2 = cpt == ' ' ? s.flags(pos + 1) : flags;
            if (!flags2.isWhitespace() && !flags2.isLetter() && !flags2.isAccentMark() && !flags2.isNumber()
                    && flags.hasCategory()) {
                pos += cpt == ' ' ? 1 : 0;
                UnicodeCptFlags f = s.flags(pos);
                while (!f.isWhitespace() && !f.isLetter() && !f.isAccentMark() && !f.isNumber() && f.hasCategory()) {
                    f = s.flags(++pos);
                }
                int cpt2 = s.cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = s.cpt(++pos);
                }
                s.addToken(pos);
                return pos;
            }
            return whitespaceAndSymbolTail(s, pos, cpt, flags);
        }
    }

    private static final class KimiK2Runner implements Runner {
        public void run(Segment s) {
            s.run(KimiK2Runner::step);
        }

        private static int step(Segment s, int pos) {
            int cpt = s.cpt(pos);
            UnicodeCptFlags flags = s.flags(pos);

            if (UnicodeCptFlags.isHan(cpt)) {
                while (UnicodeCptFlags.isHan(s.cpt(pos))) {
                    pos++;
                }
                s.addToken(pos);
                return pos;
            }

            boolean isLetterPattern = (flags.isLetter() && !UnicodeCptFlags.isHan(cpt))
                    || (!(cpt == '\r' || cpt == '\n' || flags.isLetter() || flags.isNumber())
                            && s.flags(pos + 1).isLetter() && !UnicodeCptFlags.isHan(s.cpt(pos + 1)));

            if (isLetterPattern) {
                boolean hasLeadingChar = false;
                if (!(cpt == '\r' || cpt == '\n' || flags.isLetter() || flags.isNumber())) {
                    hasLeadingChar = true;
                    pos++;
                }

                boolean hasLetters = false;
                while (s.flags(pos).isLetter() && !UnicodeCptFlags.isHan(s.cpt(pos))) {
                    hasLetters = true;
                    pos++;
                }

                if (hasLetters
                        || (!hasLeadingChar && s.flags(pos).isLetter() && !UnicodeCptFlags.isHan(s.cpt(pos)))) {
                    if (!hasLetters) {
                        pos++;
                    }
                    while (s.flags(pos).isLetter() && !UnicodeCptFlags.isHan(s.cpt(pos))) {
                        pos++;
                    }
                    if (s.cpt(pos) == '\'' && s.cpt(pos + 1) != OUT_OF_RANGE) {
                        int next = UnicodeCpt.toLower(s.cpt(pos + 1));
                        if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                            pos += 2;
                        } else if (s.cpt(pos + 2) != OUT_OF_RANGE) {
                            int n2 = UnicodeCpt.toLower(s.cpt(pos + 2));
                            if ((next == 'r' && n2 == 'e') || (next == 'v' && n2 == 'e') || (next == 'l' && n2 == 'l')) {
                                pos += 3;
                            }
                        }
                    }
                    s.addToken(pos);
                    return pos;
                }
                if (hasLeadingChar) {
                    pos--;
                }
            }

            if (flags.isNumber()) {
                int ini = pos;
                while (s.flags(pos).isNumber()) {
                    if (++pos - ini >= 3) {
                        s.addToken(pos);
                        ini = pos;
                    }
                }
                s.addToken(pos);
                return pos;
            }

            UnicodeCptFlags flags2 = cpt == ' ' ? s.flags(pos + 1) : flags;
            if (!flags2.isWhitespace() && !flags2.isLetter() && !flags2.isNumber() && flags2.hasCategory()) {
                pos += cpt == ' ' ? 1 : 0;
                UnicodeCptFlags f = s.flags(pos);
                while (!f.isWhitespace() && !f.isLetter() && !f.isNumber() && f.hasCategory()) {
                    f = s.flags(++pos);
                }
                int cpt2 = s.cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = s.cpt(++pos);
                }
                s.addToken(pos);
                return pos;
            }

            return whitespaceAndSymbolTail(s, pos, cpt, flags);
        }
    }

    private static int whitespaceAndSymbolTail(Segment s, int pos, int cpt, UnicodeCptFlags flags) {
        UnicodeCptFlags flags2 = cpt == ' ' ? s.flags(pos + 1) : flags;
        if (!flags2.isWhitespace() && !flags2.isLetter() && !flags2.isNumber() && flags.hasCategory()) {
            pos += cpt == ' ' ? 1 : 0;
            UnicodeCptFlags f = s.flags(pos);
            while (!f.isWhitespace() && !f.isLetter() && !f.isNumber() && f.hasCategory()) {
                f = s.flags(++pos);
            }
            int cpt2 = s.cpt(pos);
            while (cpt2 == '\r' || cpt2 == '\n') {
                cpt2 = s.cpt(++pos);
            }
            s.addToken(pos);
            return pos;
        }
        WhitespaceScan ws = countWhitespaceWithNewlines(s, pos);
        if (ws.lastEndRn > 0) {
            s.addToken(ws.lastEndRn);
            return ws.lastEndRn;
        }
        if (ws.count > 1 && s.cpt(pos + ws.count) != OUT_OF_RANGE) {
            s.addToken(pos + ws.count - 1);
            return pos + ws.count - 1;
        }
        if (ws.count > 0) {
            s.addToken(pos + ws.count);
            return pos + ws.count;
        }
        s.addToken(pos + 1);
        return pos + 1;
    }

    private static int countWhitespace(Segment s, int pos) {
        int count = 0;
        while (s.flags(pos + count).isWhitespace()) {
            count++;
        }
        return count;
    }

    private static WhitespaceScan countWhitespaceWithNewlines(Segment s, int pos) {
        int count = 0;
        int lastEndRn = 0;
        while (s.flags(pos + count).isWhitespace()) {
            int cpt2 = s.cpt(pos + count);
            if (cpt2 == '\r' || cpt2 == '\n') {
                lastEndRn = pos + count + 1;
            }
            count++;
        }
        return new WhitespaceScan(count, lastEndRn);
    }

    private record WhitespaceScan(int count, int lastEndRn) {
    }
}
