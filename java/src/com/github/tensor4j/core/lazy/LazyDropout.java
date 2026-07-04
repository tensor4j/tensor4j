/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core.lazy;

import com.github.tensor4j.core.DropoutMath;
import com.github.tensor4j.core.Tensor;

/** Dropout as {@code input * mask * scale} (backward via existing lazy rules). */
final class LazyDropout {

    private LazyDropout() {
    }

    static LazyUOp dropout(LazyUOp input, LazyUOp mask, float p) {
        float scale = DropoutMath.scale(p);
        LazyUOp scaledMask = LazyUOp.binary(LazyUOp.Kind.MUL, mask, LazyUOp.buffer(Tensor.of(scale)));
        return LazyUOp.binary(LazyUOp.Kind.MUL, input, scaledMask);
    }
}
