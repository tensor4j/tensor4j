/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core;

import java.util.Random;

/** Seedable RNG for reproducible dropout masks in tests and training samples. */
public final class TensorRng {

    private static final ThreadLocal<Random> DEFAULT = ThreadLocal.withInitial(() -> new Random(0L));

    private TensorRng() {
    }

    public static Random threadLocal() {
        return DEFAULT.get();
    }

    public static Random seeded(long seed) {
        return new Random(seed);
    }

    public static void seedDefault(long seed) {
        DEFAULT.set(new Random(seed));
    }
}
