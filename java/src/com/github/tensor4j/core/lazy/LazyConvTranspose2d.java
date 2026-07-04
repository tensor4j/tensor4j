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

/** Lazy conv_transpose2d (monolithic {@code CONV_TRANSPOSE2D} UOp). */
final class LazyConvTranspose2d {

    private LazyConvTranspose2d() {
    }

    static LazyUOp convTranspose2d(LazyUOp input, LazyUOp weight, int[] arg) {
        return LazyUOp.binary(LazyUOp.Kind.CONV_TRANSPOSE2D, input, weight, arg);
    }
}
