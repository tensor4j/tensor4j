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

import com.github.tensor4j.runtime.unicode.UnicodeRegexSplit;
import java.util.List;

/** Facade selecting Java-regex or llama.cpp unicode BPE pre-split. */
final class BpePreSplit {

    private BpePreSplit() {
    }

    static List<String> split(String text, BpePreType preType, BpeSplitMode mode) {
        if (preType == BpePreType.LLAMA_SPM || preType.regexes().length == 0) {
            if (text.isEmpty()) {
                return List.of();
            }
            return List.of(text);
        }
        if (mode == BpeSplitMode.LLAMA_UNICODE) {
            return UnicodeRegexSplit.split(text, preType.regexes());
        }
        return BpeRegexSplit.split(text, preType);
    }
}
