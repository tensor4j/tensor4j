/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GgufWeightLayoutModeTest {

    @Test
    void defaultIsTinygradTranspose() {
        assertEquals(GgufWeightLayoutMode.TINYGRAD, GgufWeightLayoutMode.parseName("tinygrad"));
        assertTrue(GgufWeightLayoutMode.TINYGRAD.transposeReverseGgufDims());
    }

    @Test
    void metadataOnlySkipsTranspose() {
        assertEquals(GgufWeightLayoutMode.METADATA_ONLY, GgufWeightLayoutMode.parseName("metadata"));
        assertTrue(!GgufWeightLayoutMode.METADATA_ONLY.transposeReverseGgufDims());
    }
}
