/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core.lazy;

/** Lazy GGUF load graph nodes (tinygrad {@code ggml_data_to_tensor}). */
final class LazyGgufLoad {

    private LazyGgufLoad() {
    }

    static LazyUOp fromSlice(LazyGgufSlice slice) {
        return switch (slice.typeId()) {
            case LazyGgufSlice.GGML_TYPE_Q4_0 -> LazyUOpCache.intern(
                    LazyUOp.Kind.DEQUANT_Q4_0, new LazyUOp[0], null, null, slice);
            case LazyGgufSlice.GGML_TYPE_F32 -> LazyUOpCache.intern(
                    LazyUOp.Kind.MMAP_F32, new LazyUOp[0], null, null, slice);
            default -> throw new IllegalArgumentException("unsupported ggml type id " + slice.typeId());
        };
    }

    static LazyUOp q4_0(LazyGgufSlice slice) {
        if (slice.typeId() != LazyGgufSlice.GGML_TYPE_Q4_0) {
            throw new IllegalArgumentException("expected Q4_0 slice");
        }
        return fromSlice(slice);
    }
}
