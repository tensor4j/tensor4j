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

import java.util.Arrays;

/** ggml tensor shape ({@code ne[GGML_MAX_DIMS]} — up to 4 dimensions). */
public final class GgmlTensorShape {

    public static final int MAX_DIMS = 4;

    private final long[] ne;

    public GgmlTensorShape(long[] ne) {
        if (ne.length != MAX_DIMS) {
            throw new IllegalArgumentException("expected " + MAX_DIMS + " dimensions");
        }
        this.ne = ne.clone();
    }

    public static GgmlTensorShape of(long d0) {
        return new GgmlTensorShape(new long[] {d0, 1, 1, 1});
    }

    public static GgmlTensorShape of(long d0, long d1) {
        return new GgmlTensorShape(new long[] {d0, d1, 1, 1});
    }

    public static GgmlTensorShape of(long d0, long d1, long d2) {
        return new GgmlTensorShape(new long[] {d0, d1, d2, 1});
    }

    public static GgmlTensorShape of(long d0, long d1, long d2, long d3) {
        return new GgmlTensorShape(new long[] {d0, d1, d2, d3});
    }

    public static GgmlTensorShape fromGguf(int nDims, long[] readDims) {
        if (nDims < 1 || nDims > MAX_DIMS) {
            throw new IllegalArgumentException("invalid n_dims " + nDims);
        }
        long[] ne = new long[] {1, 1, 1, 1};
        for (int i = 0; i < nDims; i++) {
            ne[i] = readDims[i];
            if (ne[i] < 0) {
                throw new IllegalArgumentException("negative dimension " + ne[i]);
            }
        }
        return new GgmlTensorShape(ne);
    }

    public long[] ne() {
        return ne.clone();
    }

    public long ne(int axis) {
        return ne[axis];
    }

    public int rank() {
        int rank = MAX_DIMS;
        while (rank > 1 && ne[rank - 1] == 1) {
            rank--;
        }
        return rank;
    }

    public long numElements() {
        long total = 1;
        for (long dim : ne) {
            total *= dim;
        }
        return total;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof GgmlTensorShape shape && Arrays.equals(ne, shape.ne);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(ne);
    }

    @Override
    public String toString() {
        return Arrays.toString(ne);
    }
}
