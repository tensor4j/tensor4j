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
 * ggml / GGUF inference runtime (CPU, RAM, chat-scale weights).
 *
 * <p>Study reference: {@code vendor/llama.cpp/ggml/} (gitignored local clone).
 *
 * <p><strong>Boundary:</strong> independent of {@code com.github.tensor4j.core} math.
 * Do not import runtime packages from tinygrad-track code. Algebra ({@code models.algebra})
 * stays on the tinygrad track.
 */
package com.github.tensor4j.runtime;
