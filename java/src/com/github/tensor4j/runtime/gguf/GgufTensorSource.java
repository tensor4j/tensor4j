/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.gguf;

/** Byte source for GGUF weight loading (heap file or mmap). */
public interface GgufTensorSource {

    GgufHeader header();

    byte[] tensorBytes(String name);

    GgufTensorSlice tensorSlice(String name);
}
