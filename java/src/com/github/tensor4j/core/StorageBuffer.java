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

/**
 * Flat float32 allocation with optional base/offset views (tinygrad {@code Buffer} subset).
 * <p>
 * Root buffers own {@code data}; views share the root allocation and add an element offset.
 */
public final class StorageBuffer {

    private final float[] data;
    private final StorageBuffer base;
    private final int offset;
    private final int size;

    private StorageBuffer(float[] data, StorageBuffer base, int offset, int size) {
        this.data = data;
        this.base = base;
        this.offset = offset;
        this.size = size;
    }

    public static StorageBuffer allocate(int numel) {
        if (numel < 0) {
            throw new IllegalArgumentException("numel must be non-negative");
        }
        return new StorageBuffer(new float[numel], null, 0, numel);
    }

    public static StorageBuffer wrapOwned(float[] data) {
        return new StorageBuffer(data, null, 0, data.length);
    }

    public StorageBuffer view(int elementOffset, int numel) {
        int absolute = absoluteOffset() + elementOffset;
        if (elementOffset < 0 || numel < 0 || absolute + numel > data.length) {
            throw new IndexOutOfBoundsException("buffer view out of range");
        }
        return new StorageBuffer(data, root(), absolute - root().offset, numel);
    }

    public StorageBuffer root() {
        return base == null ? this : base.root();
    }

    public int absoluteOffset() {
        return base == null ? offset : base.absoluteOffset() + offset;
    }

    public float[] data() {
        return data;
    }

    public int offset() {
        return absoluteOffset();
    }

    public int size() {
        return size;
    }

    public boolean isRoot() {
        return base == null;
    }

    /** Root allocation covering {@code data} from element 0 with no parent view. */
    public boolean isOwnedRoot() {
        return isRoot() && offset == 0 && size == data.length;
    }

    public StorageBuffer copy() {
        float[] copy = new float[size];
        System.arraycopy(data, offset(), copy, 0, size);
        return wrapOwned(copy);
    }
}
