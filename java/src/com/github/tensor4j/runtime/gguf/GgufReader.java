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
import com.github.tensor4j.runtime.ggml.GgmlType;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Reads GGUF metadata and full files (gguf.cpp {@code gguf_init_from_reader}). */
public final class GgufReader {

    private GgufReader() {
    }

    public static GgufHeader readMetadata(byte[] data) {
        return readFile(data).header();
    }

    public static GgufHeader readHeader(ByteBuffer buffer) {
        buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int mark = buffer.position();
        GgufBinary.expectMagic(buffer);
        int version = buffer.getInt();
        validateVersion(version);
        long nTensors = buffer.getLong();
        long nKv = buffer.getLong();
        List<GgufKvEntry> kv = readKv(buffer, nKv);
        int alignment = resolveAlignment(kv);
        List<GgufTensorInfo> tensors = readTensors(buffer, nTensors);
        long dataOffset = GgmlLayout.pad(buffer.position(), alignment);
        buffer.position(mark);
        return new GgufHeader(version, alignment, dataOffset, kv, tensors);
    }

    public static GgufFile readFile(byte[] data) {
        ByteBuffer buffer = GgufBinary.wrapRead(data);
        GgufBinary.expectMagic(buffer);
        int version = buffer.getInt();
        validateVersion(version);
        long nTensors = buffer.getLong();
        long nKv = buffer.getLong();
        List<GgufKvEntry> kv = readKv(buffer, nKv);
        int alignment = resolveAlignment(kv);
        List<GgufTensorInfo> tensors = readTensors(buffer, nTensors);
        long dataOffset = GgmlLayout.pad(buffer.position(), alignment);
        GgufHeader header = new GgufHeader(version, alignment, dataOffset, kv, tensors);
        return new GgufFile(header, data);
    }

    private static void validateVersion(int version) {
        if (version == 0 || version == 1) {
            throw new IllegalArgumentException("unsupported GGUF version " + version);
        }
        if (version > GgufConstants.VERSION) {
            throw new IllegalArgumentException("GGUF version " + version + " > supported "
                    + GgufConstants.VERSION);
        }
    }

    private static int resolveAlignment(List<GgufKvEntry> kv) {
        for (GgufKvEntry entry : kv) {
            if (GgufConstants.KEY_GENERAL_ALIGNMENT.equals(entry.key())
                    && entry.type() == GgufType.UINT32) {
                return ((Number) entry.value()).intValue();
            }
        }
        return GgufConstants.DEFAULT_ALIGNMENT;
    }

    private static List<GgufKvEntry> readKv(ByteBuffer buffer, long nKv) {
        List<GgufKvEntry> kv = new ArrayList<>((int) nKv);
        Set<String> keys = new HashSet<>();
        for (long i = 0; i < nKv; i++) {
            String key = GgufBinary.readString(buffer);
            if (!keys.add(key)) {
                throw new IllegalArgumentException("duplicate kv key " + key);
            }
            GgufType type = GgufType.fromId(buffer.getInt());
            Object value = GgufBinary.readValueBody(buffer, type);
            kv.add(new GgufKvEntry(key, type, value));
        }
        return kv;
    }

    private static List<GgufTensorInfo> readTensors(ByteBuffer buffer, long nTensors) {
        List<GgufTensorInfo> tensors = new ArrayList<>((int) nTensors);
        Set<String> names = new HashSet<>();
        for (long i = 0; i < nTensors; i++) {
            String name = GgufBinary.readString(buffer);
            if (!names.add(name)) {
                throw new IllegalArgumentException("duplicate tensor name " + name);
            }
            int nDims = buffer.getInt();
            if (nDims < 1 || nDims > 4) {
                throw new IllegalArgumentException("invalid n_dims " + nDims + " for " + name);
            }
            long[] dims = new long[4];
            for (int d = 0; d < nDims; d++) {
                dims[d] = buffer.getLong();
            }
            GgmlType type = GgmlType.fromId(buffer.getInt());
            long offset = buffer.getLong();
            tensors.add(GgufTensorInfo.parse(name, nDims, dims, type, offset));
        }
        return tensors;
    }
}
