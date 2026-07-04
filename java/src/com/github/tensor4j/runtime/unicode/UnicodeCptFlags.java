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

/** Unicode category flags ({@code unicode_cpt_flags} in llama.cpp unicode.cpp). */
public final class UnicodeCptFlags {

    public static final UnicodeCptFlags UNDEFINED = new UnicodeCptFlags(false, false, false, false, false, false);

    private final boolean letter;
    private final boolean number;
    private final boolean punctuation;
    private final boolean accentMark;
    private final boolean symbol;
    private final boolean whitespace;

    private UnicodeCptFlags(
            boolean letter,
            boolean number,
            boolean punctuation,
            boolean accentMark,
            boolean symbol,
            boolean whitespace) {
        this.letter = letter;
        this.number = number;
        this.punctuation = punctuation;
        this.accentMark = accentMark;
        this.symbol = symbol;
        this.whitespace = whitespace;
    }

    public boolean isLetter() {
        return letter;
    }

    public boolean isNumber() {
        return number;
    }

    public boolean isPunctuation() {
        return punctuation;
    }

    public boolean isAccentMark() {
        return accentMark;
    }

    public boolean isSymbol() {
        return symbol;
    }

    public boolean isWhitespace() {
        return whitespace;
    }

    /** True when any category bit is set (llama {@code as_uint()}). */
    public boolean hasCategory() {
        return letter || number || punctuation || accentMark || symbol || whitespace;
    }

    public static UnicodeCptFlags fromCodepoint(int cpt) {
        if (isLlamaWhitespace(cpt)) {
            return new UnicodeCptFlags(false, false, false, false, false, true);
        }
        int type = Character.getType(cpt);
        boolean letter = type == Character.UPPERCASE_LETTER
                || type == Character.LOWERCASE_LETTER
                || type == Character.TITLECASE_LETTER
                || type == Character.MODIFIER_LETTER
                || type == Character.OTHER_LETTER;
        boolean number = type == Character.DECIMAL_DIGIT_NUMBER
                || type == Character.LETTER_NUMBER
                || type == Character.OTHER_NUMBER;
        boolean punctuation = type == Character.CONNECTOR_PUNCTUATION
                || type == Character.DASH_PUNCTUATION
                || type == Character.START_PUNCTUATION
                || type == Character.END_PUNCTUATION
                || type == Character.INITIAL_QUOTE_PUNCTUATION
                || type == Character.FINAL_QUOTE_PUNCTUATION
                || type == Character.OTHER_PUNCTUATION;
        boolean accentMark = type == Character.NON_SPACING_MARK
                || type == Character.COMBINING_SPACING_MARK
                || type == Character.ENCLOSING_MARK;
        boolean symbol = type == Character.MATH_SYMBOL
                || type == Character.CURRENCY_SYMBOL
                || type == Character.MODIFIER_SYMBOL
                || type == Character.OTHER_SYMBOL;
        return new UnicodeCptFlags(letter, number, punctuation, accentMark, symbol, false);
    }

    /** llama.cpp {@code \\s}: {@code \\r\\n\\t\\f\\v} and space. */
    public static boolean isLlamaWhitespace(int cpt) {
        return cpt == ' ' || cpt == '\t' || cpt == '\n' || cpt == '\r' || cpt == '\f' || cpt == '\u000B';
    }

    /** CJK unified ideographs ({@code unicode_cpt_is_han} in llama.cpp). */
    public static boolean isHan(int cpt) {
        if (cpt >= 0x4E00 && cpt <= 0x9FFF) {
            return true;
        }
        if (cpt >= 0x3400 && cpt <= 0x4DBF) {
            return true;
        }
        if (cpt >= 0x20000 && cpt <= 0x2A6DF) {
            return true;
        }
        if (cpt >= 0x2A700 && cpt <= 0x2B73F) {
            return true;
        }
        if (cpt >= 0x2B740 && cpt <= 0x2B81F) {
            return true;
        }
        if (cpt >= 0x2B820 && cpt <= 0x2CEAF) {
            return true;
        }
        if (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) {
            return true;
        }
        if (cpt >= 0xF900 && cpt <= 0xFAFF) {
            return true;
        }
        if (cpt >= 0x2F800 && cpt <= 0x2FA1F) {
            return true;
        }
        return false;
    }
}
