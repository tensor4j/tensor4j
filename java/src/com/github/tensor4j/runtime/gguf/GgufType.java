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

/** GGUF KV value types ({@code enum gguf_type} in gguf.h). */
public enum GgufType {
    UINT8(0),
    INT8(1),
    UINT16(2),
    INT16(3),
    UINT32(4),
    INT32(5),
    FLOAT32(6),
    BOOL(7),
    STRING(8),
    ARRAY(9),
    UINT64(10),
    INT64(11),
    FLOAT64(12);

    private final int id;

    GgufType(int id) {
        this.id = id;
    }

    public int id() {
        return id;
    }

    public static GgufType fromId(int id) {
        for (GgufType type : values()) {
            if (type.id == id) {
                return type;
            }
        }
        throw new IllegalArgumentException("unknown gguf type id " + id);
    }
}
