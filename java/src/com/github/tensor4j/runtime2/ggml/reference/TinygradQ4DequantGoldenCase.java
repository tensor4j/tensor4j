/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime2.ggml.reference;

/** One Q4_0 dequant golden vector (shape + quant bytes → float[]). */
public record TinygradQ4DequantGoldenCase(
        String name,
        int[] shape,
        byte[] quantBytes,
        float[] expected,
        float tolerance) {
}
