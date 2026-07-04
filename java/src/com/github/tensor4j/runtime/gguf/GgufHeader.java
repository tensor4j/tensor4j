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

import java.util.List;

/** Parsed GGUF header (metadata only — no weight blob loaded). */
public record GgufHeader(int version, int alignment, long dataOffset, List<GgufKvEntry> kv,
        List<GgufTensorInfo> tensors) {

    public GgufKvEntry findKv(String key) {
        for (GgufKvEntry entry : kv) {
            if (entry.key().equals(key)) {
                return entry;
            }
        }
        return null;
    }

    public GgufTensorInfo findTensor(String name) {
        for (GgufTensorInfo tensor : tensors) {
            if (tensor.name().equals(name)) {
                return tensor;
            }
        }
        return null;
    }
}
