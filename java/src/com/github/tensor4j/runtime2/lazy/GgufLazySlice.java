/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime2.lazy;

import com.github.tensor4j.core.lazy.LazyGgufSlice;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import com.github.tensor4j.runtime.gguf.GgufTensorSlice;

/** Bridge GGUF mmap slices into core {@link LazyGgufSlice} (tinygrad {@code ggml_data_to_tensor}). */
public final class GgufLazySlice {

    private GgufLazySlice() {
    }

    public static LazyGgufSlice from(GgufTensorSlice slice) {
        int[] dims = floatDims(slice.shape());
        if (slice.type() == GgmlType.Q4_0) {
            return LazyGgufSlice.q4_0(slice.buffer(), slice.offset(), dims);
        }
        if (slice.type() == GgmlType.F32) {
            return LazyGgufSlice.f32(slice.buffer(), slice.offset(), dims);
        }
        throw new IllegalArgumentException("lazy load supports F32 and Q4_0, got " + slice.type());
    }

    public static int[] floatDims(GgmlTensorShape shape) {
        int rank = shape.rank();
        long[] ne = shape.ne();
        int[] dims = new int[rank];
        for (int i = 0; i < rank; i++) {
            dims[i] = (int) ne[i];
        }
        return dims;
    }
}
