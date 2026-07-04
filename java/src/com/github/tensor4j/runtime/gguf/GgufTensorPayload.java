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

import com.github.tensor4j.runtime.ggml.GgmlLayout;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;

/** Raw tensor weight bytes for GGUF file writing. */
public record GgufTensorPayload(String name, GgmlType type, GgmlTensorShape shape, byte[] data) {

    public GgufTensorPayload {
        long expected = GgmlLayout.numBytes(type, shape);
        if (data.length != expected) {
            throw new IllegalArgumentException("data length " + data.length + " != expected " + expected
                    + " for " + name);
        }
    }
}
