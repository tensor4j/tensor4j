/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
/**
 * tinygrad-aligned inference runtime (lazy UOp DAG, state_dict, gguf_load).
 *
 * <p>Study reference: tinygrad {@code ggml_data_to_tensor}, {@code load_state_dict},
 * {@code Tensor.realize()}.
 *
 * <p><strong>Boundary:</strong> do not modify {@code com.github.tensor4j.runtime} (llama.cpp track).
 * Shared GGUF mmap and {@link com.github.tensor4j.runtime.infer.InferTensor} math may be reused;
 * weight loading and realize semantics live here.
 */
package com.github.tensor4j.runtime2;
