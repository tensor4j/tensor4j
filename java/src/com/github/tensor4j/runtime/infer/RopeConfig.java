/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.infer;

/** RoPE config (ggml_compute_forward_rope + YaRN, llama hparams). */
public record RopeConfig(
        float freqBase,
        float freqScale,
        int ropeDim,
        RopeScalingType scaling,
        float yarnExtFactor,
        float yarnAttnFactor,
        float yarnBetaFast,
        float yarnBetaSlow,
        int yarnOrigCtx) {

    public static RopeConfig llamaDefaults(int headDim) {
        return new RopeConfig(10000.0f, 1.0f, headDim, RopeScalingType.NONE, 0.0f, 1.0f, 32.0f, 1.0f, 0);
    }

    public static RopeConfig disabled() {
        return new RopeConfig(10000.0f, 1.0f, 0, RopeScalingType.NONE, 0.0f, 1.0f, 32.0f, 1.0f, 0);
    }

    public boolean enabled() {
        return ropeDim > 0;
    }

    public boolean yarnEnabled() {
        return scaling == RopeScalingType.YARN && yarnExtFactor != 0.0f;
    }
}
