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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Parsed GGUF file with metadata and weight blob. */
public record GgufFile(GgufHeader header, byte[] bytes) implements GgufTensorSource {

    @Override
    public byte[] tensorBytes(String name) {
        GgufTensorSlice slice = tensorSlice(name);
        byte[] out = new byte[slice.sizeBytes()];
        slice.buffer().position(slice.offset());
        slice.buffer().get(slice.offset(), out);
        return out;
    }

    @Override
    public GgufTensorSlice tensorSlice(String name) {
        GgufTensorInfo tensor = header.findTensor(name);
        if (tensor == null) {
            throw new IllegalArgumentException("unknown tensor " + name);
        }
        long absolute = header.dataOffset() + tensor.offsetBytes();
        if (absolute + tensor.sizeBytes() > bytes.length) {
            throw new IllegalArgumentException("tensor " + name + " extends past file end");
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        return new GgufTensorSlice(buffer, (int) absolute, tensor);
    }
}
