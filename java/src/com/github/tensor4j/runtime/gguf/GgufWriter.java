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
import java.nio.ByteBuffer;
import java.util.List;

/** Writes GGUF metadata and full files (gguf.cpp layout). */
public final class GgufWriter {

    private GgufWriter() {
    }

    public static byte[] writeMetadata(int version, List<GgufKvEntry> kv, List<GgufTensorInfo> tensors) {
        return writeMetadataSection(version, kv, tensors, resolveAlignment(kv));
    }

    /** Write metadata + aligned weight data section (gguf.cpp write path). */
    public static byte[] writeFile(int version, List<GgufKvEntry> kv, List<GgufTensorPayload> payloads) {
        int alignment = resolveAlignment(kv);
        List<GgufTensorInfo> tensors = GgufWeightLayout.plan(payloads, alignment);
        byte[] metadata = writeMetadataSection(version, kv, tensors, alignment);
        long dataBytes = GgufWeightLayout.totalDataBytes(tensors, alignment);
        byte[] dataSection = new byte[(int) dataBytes];
        for (int i = 0; i < payloads.size(); i++) {
            GgufTensorInfo info = tensors.get(i);
            System.arraycopy(payloads.get(i).data(), 0, dataSection, (int) info.offsetBytes(),
                    (int) info.sizeBytes());
        }
        byte[] file = new byte[metadata.length + dataSection.length];
        System.arraycopy(metadata, 0, file, 0, metadata.length);
        System.arraycopy(dataSection, 0, file, metadata.length, dataSection.length);
        return file;
    }

    private static byte[] writeMetadataSection(int version, List<GgufKvEntry> kv, List<GgufTensorInfo> tensors,
            int alignment) {
        int capacity = estimateCapacity(kv, tensors);
        ByteBuffer buffer = GgufBinary.allocateWrite(capacity);
        GgufBinary.writeMagic(buffer);
        buffer.putInt(version);
        buffer.putLong(tensors.size());
        buffer.putLong(kv.size());
        for (GgufKvEntry entry : kv) {
            GgufBinary.writeString(buffer, entry.key());
            GgufBinary.writeValue(buffer, entry.type(), entry.value());
        }
        for (GgufTensorInfo tensor : tensors) {
            writeTensorInfo(buffer, tensor);
        }
        while (buffer.position() % alignment != 0) {
            buffer.put((byte) 0);
        }
        byte[] out = new byte[buffer.position()];
        buffer.flip();
        buffer.get(out);
        return out;
    }

    private static void writeTensorInfo(ByteBuffer buffer, GgufTensorInfo tensor) {
        GgufBinary.writeString(buffer, tensor.name());
        long[] ne = tensor.shape().ne();
        int nDims = tensor.shape().rank();
        buffer.putInt(nDims);
        for (int d = 0; d < nDims; d++) {
            buffer.putLong(ne[d]);
        }
        buffer.putInt(tensor.type().id());
        buffer.putLong(tensor.offsetBytes());
    }

    private static int resolveAlignment(List<GgufKvEntry> kv) {
        for (GgufKvEntry entry : kv) {
            if (GgufConstants.KEY_GENERAL_ALIGNMENT.equals(entry.key())) {
                return ((Number) entry.value()).intValue();
            }
        }
        return GgufConstants.DEFAULT_ALIGNMENT;
    }

    private static int estimateCapacity(List<GgufKvEntry> kv, List<GgufTensorInfo> tensors) {
        int size = 64;
        for (GgufKvEntry entry : kv) {
            size += entry.key().length() + 32;
        }
        size += tensors.size() * 128;
        return size;
    }

    /** Build a tensor info record with computed size and strides. */
    public static GgufTensorInfo tensor(String name, GgmlType type, GgmlTensorShape shape, long offsetBytes) {
        long sizeBytes = GgmlLayout.numBytes(type, shape);
        long[] byteStrides = GgmlLayout.byteStrides(type, shape);
        return new GgufTensorInfo(name, shape, type, offsetBytes, sizeBytes, byteStrides);
    }
}
