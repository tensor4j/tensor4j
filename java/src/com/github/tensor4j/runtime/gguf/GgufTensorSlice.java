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

import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import java.nio.ByteBuffer;

/** Zero-copy view into a GGUF weight blob (mmap-backed or heap). */
public record GgufTensorSlice(ByteBuffer buffer, int offset, GgufTensorInfo info) {

    public GgufTensorSlice {
        if (buffer.order() != java.nio.ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException("buffer must be little-endian");
        }
    }

    public GgmlType type() {
        return info.type();
    }

    public GgmlTensorShape shape() {
        return info.shape();
    }

    public String name() {
        return info.name();
    }

    public int sizeBytes() {
        return (int) info.sizeBytes();
    }
}
