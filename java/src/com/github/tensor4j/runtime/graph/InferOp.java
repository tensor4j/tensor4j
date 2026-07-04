/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.graph;

/** Inference graph op kinds (ggml graph nodes, forward-only). */
public enum InferOp {
    INPUT,
    RMS_NORM,
    MUL_MAT,
    ADD,
    SILU,
    SWIGLU,
    MUL
}
