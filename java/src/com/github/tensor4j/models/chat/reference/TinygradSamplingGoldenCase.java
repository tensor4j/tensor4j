/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.reference;

/** One golden sampling row: 1d logits in, scalar token id out. */
public record TinygradSamplingGoldenCase(
        String name,
        float[] logits,
        float temperature,
        long seed,
        double multinomialRoll,
        int topK,
        float topP,
        float alphaFrequency,
        float alphaPresence,
        int[] alphaCounts,
        boolean gumbelMax,
        int expectedToken) {
}
