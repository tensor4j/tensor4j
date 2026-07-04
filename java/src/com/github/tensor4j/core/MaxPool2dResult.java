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

/** Result of max-pool with spatial indices (tinygrad {@code return_indices=True}). */
public final class MaxPool2dResult {

    public final Tensor output;
    /** Per-output flat spatial index into {@code H x W} (stored as float). */
    public final Tensor indices;

    public MaxPool2dResult(Tensor output, Tensor indices) {
        this.output = output;
        this.indices = indices;
    }
}
