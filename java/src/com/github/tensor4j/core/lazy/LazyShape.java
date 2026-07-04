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

import com.github.tensor4j.core.Strides;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.core.TensorLayout;
import java.util.Arrays;

/**
 * Shape-only lazy graph (tinygrad {@code UOp._shape} before {@code realize()}).
 * Movement metadata is computed without allocating a buffer.
 */
public final class LazyShape {

    /** Mirrors tinygrad {@code GroupOp.Movement} plus reduce/broadcast shape rules. */
    public enum Kind {
        LEAF,
        RESHAPE,
        PERMUTE,
        EXPAND,
        REDUCE
    }

    private final Kind kind;
    private final LazyShape src;
    private final int[] arg;
    private int[] cachedShape;

    private LazyShape(Kind kind, LazyShape src, int[] arg) {
        this.kind = kind;
        this.src = src;
        this.arg = arg == null ? null : arg.clone();
    }

    public static LazyShape leaf(int... shape) {
        validatePositiveShape(shape);
        return new LazyShape(Kind.LEAF, null, shape);
    }

    public Kind kind() {
        return kind;
    }

    public LazyShape source() {
        return src;
    }

    /** Resolved shape (tinygrad {@code UOp.shape} / {@code _shape}). */
    public int[] shape() {
        if (cachedShape != null) {
            return cachedShape.clone();
        }
        cachedShape = computeShape();
        return cachedShape.clone();
    }

    public int numel() {
        return Strides.numel(shape());
    }

    public int rank() {
        return shape().length;
    }

    public LazyShape reshape(int... newShape) {
        return new LazyShape(Kind.RESHAPE, this, newShape);
    }

    public LazyShape permute(int... order) {
        return new LazyShape(Kind.PERMUTE, this, order);
    }

    /** tinygrad {@code expand}: source and target must have the same rank. */
    public LazyShape expand(int... targetShape) {
        return new LazyShape(Kind.EXPAND, this, targetShape);
    }

    /** Collapse one axis (tinygrad {@code REDUCE} shape rule). */
    public LazyShape reduceAxis(int axis) {
        return new LazyShape(Kind.REDUCE, this, new int[] {axis});
    }

    /** Elementwise broadcast shape (tinygrad {@code _broadcast_shape}). */
    public static int[] broadcastShape(int[] left, int[] right) {
        return BroadcastShape.compute(left, right);
    }

    public static LazyShape broadcast(LazyShape left, LazyShape right) {
        int[] outShape = broadcastShape(left.shape(), right.shape());
        return LazyShape.leaf(outShape);
    }

    /** Walk movement chain and apply to an eager layout (post-{@code realize} view). */
    public TensorLayout toLayout(TensorLayout base) {
        if (!Arrays.equals(base.shape(), leafShape())) {
            throw new IllegalArgumentException("base layout " + Arrays.toString(base.shape())
                    + " != lazy leaf " + Arrays.toString(leafShape()));
        }
        return applyLayout(this, base);
    }

    /** Apply the same movement chain on a concrete {@link Tensor}. */
    public Tensor materialize(Tensor root) {
        if (!Arrays.equals(root.shape().dims(), leafShape())) {
            throw new IllegalArgumentException("tensor shape " + root.shape()
                    + " != lazy leaf " + Arrays.toString(leafShape()));
        }
        return applyTensor(this, root);
    }

    public int[] leafShape() {
        LazyShape node = this;
        while (node.kind != Kind.LEAF) {
            node = node.src;
        }
        return node.arg.clone();
    }

    public int movementDepth() {
        int depth = 0;
        LazyShape node = this;
        while (node.kind != Kind.LEAF) {
            depth++;
            node = node.src;
        }
        return depth;
    }

    private int[] computeShape() {
        switch (kind) {
            case LEAF:
                return arg.clone();
            case RESHAPE:
                return reshapeShape(src.shape(), arg);
            case PERMUTE:
                return permuteShape(src.shape(), arg);
            case EXPAND:
                return expandShape(src.shape(), arg);
            case REDUCE:
                return reduceShape(src.shape(), arg[0]);
            default:
                throw new IllegalStateException("unhandled kind " + kind);
        }
    }

    private static int[] reshapeShape(int[] sourceShape, int[] newShape) {
        if (newShape.length == 0) {
            throw new IllegalArgumentException("reshape target cannot be empty");
        }
        for (int dim : newShape) {
            if (dim < 0) {
                throw new IllegalArgumentException("shape cannot contain negative numbers "
                        + Arrays.toString(newShape));
            }
        }
        if (Strides.numel(sourceShape) != Strides.numel(newShape)) {
            throw new IllegalArgumentException("bad reshape: " + Arrays.toString(sourceShape)
                    + " -> " + Arrays.toString(newShape));
        }
        return newShape.clone();
    }

    private static int[] permuteShape(int[] sourceShape, int[] order) {
        if (order.length != sourceShape.length) {
            throw new IllegalArgumentException("invalid permutation length " + order.length
                    + " for rank " + sourceShape.length);
        }
        int[] sorted = order.clone();
        Arrays.sort(sorted);
        for (int axis = 0; axis < sorted.length; axis++) {
            if (sorted[axis] != axis) {
                throw new IllegalArgumentException("invalid permutation " + Arrays.toString(order));
            }
        }
        int[] result = new int[order.length];
        for (int axis = 0; axis < order.length; axis++) {
            result[axis] = sourceShape[order[axis]];
        }
        return result;
    }

    private static int[] expandShape(int[] sourceShape, int[] targetShape) {
        if (sourceShape.length != targetShape.length) {
            throw new IllegalArgumentException("bad expand rank: " + Arrays.toString(sourceShape)
                    + " -> " + Arrays.toString(targetShape));
        }
        for (int axis = 0; axis < sourceShape.length; axis++) {
            int sourceDim = sourceShape[axis];
            int targetDim = targetShape[axis];
            if (sourceDim != targetDim && !(sourceDim == 1 && targetDim >= 1)) {
                throw new IllegalArgumentException("bad expand: " + Arrays.toString(sourceShape)
                        + " -> " + Arrays.toString(targetShape));
            }
        }
        return targetShape.clone();
    }

    private static int[] reduceShape(int[] sourceShape, int axis) {
        if (axis < 0 || axis >= sourceShape.length) {
            throw new IllegalArgumentException("reduce axis out of range: " + axis);
        }
        int[] result = new int[sourceShape.length - 1];
        for (int i = 0, j = 0; i < sourceShape.length; i++) {
            if (i != axis) {
                result[j++] = sourceShape[i];
            }
        }
        return result;
    }

    private static TensorLayout applyLayout(LazyShape node, TensorLayout layout) {
        if (node.kind == Kind.LEAF) {
            return layout;
        }
        TensorLayout child = applyLayout(node.src, layout);
        switch (node.kind) {
            case RESHAPE:
                return child.reshape(node.arg);
            case PERMUTE:
                return child.permute(node.arg);
            case EXPAND:
                return child.expand(node.arg);
            case REDUCE:
                throw new UnsupportedOperationException("reduce materializes new storage in eager tensor4j");
            default:
                throw new IllegalStateException("unhandled kind " + node.kind);
        }
    }

    private static Tensor applyTensor(LazyShape node, Tensor tensor) {
        if (node.kind == Kind.LEAF) {
            return tensor;
        }
        Tensor child = applyTensor(node.src, tensor);
        switch (node.kind) {
            case RESHAPE:
                return child.reshape(node.arg);
            case PERMUTE:
                return child.permute(node.arg);
            case EXPAND:
                return child.expand(node.arg);
            case REDUCE:
                return child.sumAxis(node.arg[0]);
            default:
                throw new IllegalStateException("unhandled kind " + node.kind);
        }
    }

    private static void validatePositiveShape(int[] shape) {
        if (shape.length == 0) {
            throw new IllegalArgumentException("shape must have at least one dimension");
        }
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("dimensions must be positive: " + Arrays.toString(shape));
            }
        }
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof LazyShape lazyShape)) {
            return false;
        }
        return kind == lazyShape.kind && Arrays.equals(arg, lazyShape.arg) && ObjectsEqual.srcEquals(src, lazyShape.src);
    }

    @Override
    public int hashCode() {
        return kind.hashCode() ^ Arrays.hashCode(arg) ^ (src == null ? 0 : src.hashCode());
    }

    @Override
    public String toString() {
        return "LazyShape{" + kind + " shape=" + Arrays.toString(shape()) + "}";
    }

    /** Avoid requiring Java 17 Objects helper in core for null-safe src equality. */
    private static final class ObjectsEqual {
        private ObjectsEqual() {
        }

        private static boolean srcEquals(LazyShape left, LazyShape right) {
            if (left == right) {
                return true;
            }
            if (left == null || right == null) {
                return false;
            }
            return left.equals(right);
        }
    }
}
