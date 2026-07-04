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

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Memory-mapped GGUF file for bounded RAM weight access. */
public final class MmappedGgufFile implements GgufTensorSource, AutoCloseable {

    private final GgufHeader header;
    private final MappedByteBuffer buffer;
    private final FileChannel channel;

    private MmappedGgufFile(GgufHeader header, MappedByteBuffer buffer, FileChannel channel) {
        this.header = header;
        this.buffer = buffer;
        this.channel = channel;
    }

    public static MmappedGgufFile open(Path path) throws IOException {
        FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
        MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
        mapped.order(ByteOrder.LITTLE_ENDIAN);
        GgufHeader header = GgufReader.readHeader(mapped);
        return new MmappedGgufFile(header, mapped, channel);
    }

    @Override
    public GgufHeader header() {
        return header;
    }

    @Override
    public byte[] tensorBytes(String name) {
        GgufTensorSlice slice = tensorSlice(name);
        byte[] out = new byte[slice.sizeBytes()];
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
        if (absolute + tensor.sizeBytes() > buffer.capacity()) {
            throw new IllegalArgumentException("tensor " + name + " extends past file end");
        }
        return new GgufTensorSlice(buffer, (int) absolute, tensor);
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}
