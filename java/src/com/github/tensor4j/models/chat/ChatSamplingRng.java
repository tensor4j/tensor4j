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

import java.security.SecureRandom;
import java.util.Random;

/**
 * Random draws for chat sampling — tinygrad {@code apps/llm.py} and {@code extra/models/llama.py}.
 *
 * <p>Gumbel-max ({@code apps/llm.py}):
 * {@code argmax(logits/temperature - log(-log(uniform)))} where {@code uniform ~ U(0,1)} clamped to
 * {@code >= 1e-12}. Equivalent to sampling from {@code softmax(logits/temperature)}.
 *
 * <p>Default entropy is {@link SecureRandom}. Use {@link ChatSamplingRngMode#LEGACY} with
 * {@link ChatGenerationOptions#seed()} for deterministic tests.
 */
public final class ChatSamplingRng {

    public static final double MIN_UNIFORM = 1e-12;

    private final Random source;
    private final ChatSamplingRngMode mode;
    private final Double fixedMultinomialRoll;

    public static ChatSamplingRng create(ChatSamplingRngMode mode, long seed) {
        if (mode == ChatSamplingRngMode.LEGACY) {
            return new ChatSamplingRng(new Random(seed), ChatSamplingRngMode.LEGACY);
        }
        return new ChatSamplingRng(new SecureRandom(), ChatSamplingRngMode.SECURE);
    }

    public static ChatSamplingRng fromOptions(ChatGenerationOptions options) {
        return create(options.rngMode(), options.seed());
    }

    /** Legacy seeded path for unit tests. */
    public ChatSamplingRng(long seed) {
        this(new Random(seed), ChatSamplingRngMode.LEGACY);
    }

    public ChatSamplingRng(Random source) {
        this(source, source instanceof SecureRandom ? ChatSamplingRngMode.SECURE : ChatSamplingRngMode.LEGACY);
    }

    ChatSamplingRng(Random source, ChatSamplingRngMode mode) {
        this(source, mode, null);
    }

    ChatSamplingRng(Random source, ChatSamplingRngMode mode, Double fixedMultinomialRoll) {
        this.source = source;
        this.mode = mode;
        this.fixedMultinomialRoll = fixedMultinomialRoll;
    }

    public static ChatSamplingRng fixedMultinomialRoll(double roll) {
        return new ChatSamplingRng(new Random(0L), ChatSamplingRngMode.LEGACY, roll);
    }

    public ChatSamplingRngMode mode() {
        return mode;
    }

    public Random source() {
        return source;
    }

    /** Uniform on (0,1], lower-bounded — tinygrad {@code Tensor.rand_like(...).maximum(1e-12)}. */
    public double nextUniformClamped() {
        return Math.max(source.nextDouble(), MIN_UNIFORM);
    }

    /** Standard Gumbel(0,1): {@code -log(-log(u))}. */
    public static double standardGumbelFromUniform(double u) {
        double clamped = Math.max(u, MIN_UNIFORM);
        return -Math.log(-Math.log(clamped));
    }

    public double nextStandardGumbel() {
        return standardGumbelFromUniform(nextUniformClamped());
    }

    /** Gumbel-max score for one logit — tinygrad {@code logits/temp - log(-log(uniform))}. */
    public double gumbelMaxScore(float logit, float temperature) {
        float temp = Math.max(temperature, (float) MIN_UNIFORM);
        return logit / temp + nextStandardGumbel();
    }

    /** Unit interval roll for inverse-CDF multinomial sampling. */
    public double nextMultinomialRoll() {
        if (fixedMultinomialRoll != null) {
            return fixedMultinomialRoll;
        }
        return source.nextDouble();
    }
}
