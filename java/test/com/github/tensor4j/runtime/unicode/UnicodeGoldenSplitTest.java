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

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.support.TokenizerGoldenFixtures;
import com.github.tensor4j.support.TokenizerGoldenFixtures.GoldenCase;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Golden pre-split vectors ({@code unicode_regex_split} parity with llama.cpp). */
class UnicodeGoldenSplitTest {

    @Test
    void goldenCasesMatchLlamaUnicodeSplit() {
        for (GoldenCase golden : TokenizerGoldenFixtures.UNICODE_SPLIT_CASES) {
            List<String> parts = UnicodeRegexSplit.split(golden.input(), golden.regexes());
            assertEquals(
                    List.of(golden.expectedParts()),
                    parts,
                    golden.name() + " split of \"" + golden.input() + "\"");
        }
    }
}
