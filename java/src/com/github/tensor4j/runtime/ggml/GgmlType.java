/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.ggml;

/** ggml tensor element types ({@code enum ggml_type} in ggml.h). */
public enum GgmlType {
    F32(0, 4, 1),
    F16(1, 2, 1),
    Q4_0(2, 18, 32),
    Q4_1(3, 20, 32),
    Q5_0(6, 22, 32),
    Q5_1(7, 24, 32),
    Q8_0(8, 34, 32),
    Q8_1(9, 36, 32),
    I8(24, 1, 1),
    I16(25, 2, 1),
    I32(26, 4, 1),
    I64(27, 8, 1),
    F64(28, 8, 1),
    BF16(30, 2, 1);

    private final int id;
    private final int typeSizeBytes;
    private final int blockSize;

    GgmlType(int id, int typeSizeBytes, int blockSize) {
        this.id = id;
        this.typeSizeBytes = typeSizeBytes;
        this.blockSize = blockSize;
    }

    public int id() {
        return id;
    }

    /** Bytes per quantization block ({@code ggml_type_size}). */
    public int typeSizeBytes() {
        return typeSizeBytes;
    }

    /** Elements per block ({@code ggml_blck_size}). */
    public int blockSize() {
        return blockSize;
    }

    public static GgmlType fromId(int id) {
        for (GgmlType type : values()) {
            if (type.id == id) {
                return type;
            }
        }
        throw new IllegalArgumentException("unknown ggml type id " + id);
    }
}
