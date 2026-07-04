/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core;

import java.util.Arrays;

/** Immutable tensor shape, aligned with tinygrad's multi-dimensional layout. */
public final class Shape {

    private final int[] dims;

    public Shape(int... dims) {
        if (dims.length == 0) {
            throw new IllegalArgumentException("shape must have at least one dimension");
        }
        for (int dim : dims) {
            if (dim <= 0) {
                throw new IllegalArgumentException("dimensions must be positive: " + Arrays.toString(dims));
            }
        }
        this.dims = dims.clone();
    }

    public static Shape scalar() {
        return new Shape(1);
    }

    public static Shape vector(int length) {
        return new Shape(length);
    }

    public static Shape matrix(int rows, int cols) {
        return new Shape(rows, cols);
    }

    public int rank() {
        return dims.length;
    }

    public int dim(int index) {
        return dims[index];
    }

    public int[] dims() {
        return dims.clone();
    }

    public int numel() {
        int total = 1;
        for (int dim : dims) {
            total *= dim;
        }
        return total;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Shape shape && Arrays.equals(dims, shape.dims);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }

    @Override
    public String toString() {
        return Arrays.toString(dims);
    }
}
