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

final class LazyMaxUnpool2d {

    private LazyMaxUnpool2d() {
    }

    static LazyUOp maxUnpool2d(LazyUOp values, LazyUOp indices, int[] poolArg, int[] outputShape) {
        int[] packed = new int[poolArg.length + outputShape.length];
        System.arraycopy(poolArg, 0, packed, 0, poolArg.length);
        System.arraycopy(outputShape, 0, packed, poolArg.length, outputShape.length);
        return LazyUOp.multi(LazyUOp.Kind.MAX_UNPOOL2D, new LazyUOp[] {values, indices}, packed);
    }

    static int[] poolArgFromPacked(int[] packed) {
        return java.util.Arrays.copyOfRange(packed, 0, 9);
    }

    static int[] outputShapeFromPacked(int[] packed) {
        return java.util.Arrays.copyOfRange(packed, 9, 13);
    }

    static int[] valueShapeFromGradPacked(int[] packed) {
        return java.util.Arrays.copyOfRange(packed, 13, 17);
    }

    static int[] gradArg(int[] forwardPacked, int[] valueShape) {
        int[] out = new int[forwardPacked.length + valueShape.length];
        System.arraycopy(forwardPacked, 0, out, 0, forwardPacked.length);
        System.arraycopy(valueShape, 0, out, forwardPacked.length, valueShape.length);
        return out;
    }
}
