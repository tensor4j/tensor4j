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

import com.github.tensor4j.core.Tensor;
import java.util.Arrays;
import java.util.Objects;

/**
 * Lazy computation-graph node (tinygrad {@code UOp}).
 * Each node holds an op, parent nodes ({@code src}), and optional args — a DAG, not a linear op list.
 */
public final class LazyUOp {

    public enum Kind {
        BUFFER,
        RESHAPE,
        PERMUTE,
        EXPAND,
        PAD,
        SHRINK,
        CONTIGUOUS,
        CAST,
        ADD,
        MUL,
        SUB,
        DIV,
        MAX,
        WHERE,
        NEG,
        RECIP,
        POW,
        LOG2,
        EXP2,
        EXP,
        SQRT,
        RELU,
        RELU_MASK,
        GT_MASK,
        EQ_MASK,
        SUM,
        SUM_AXIS,
        MAX_AXIS,
        MEAN,
        CONV2D,
        CONV2D_INPUT_GRAD,
        CONV2D_WEIGHT_GRAD,
        IM2COL,
        IM2COL_INPUT_GRAD,
        POOL2D,
        CONV_TRANSPOSE2D,
        CONV_TRANSPOSE2D_INPUT_GRAD,
        CONV_TRANSPOSE2D_WEIGHT_GRAD,
        MAX_UNPOOL2D,
        MAX_UNPOOL2D_VALUE_GRAD,
        /** Lazy Q4_0 dequant from {@link LazyGgufSlice} (tinygrad {@code ggml_data_to_tensor}). */
        DEQUANT_Q4_0,
        /** Lazy F32 bitcast from mmap slice (tinygrad native F32 path). */
        MMAP_F32
    }

    private final Kind op;
    private final LazyUOp[] src;
    private final int[] arg;
    private final Tensor buffer;
    private final LazyGgufSlice ggufSlice;

    LazyUOp(Kind op, LazyUOp[] src, int[] arg, Tensor buffer, LazyGgufSlice ggufSlice) {
        this.op = op;
        this.src = src == null ? new LazyUOp[0] : src;
        this.arg = arg == null ? null : arg.clone();
        this.buffer = buffer;
        this.ggufSlice = ggufSlice;
    }

    static LazyUOp buffer(Tensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("buffer tensor required");
        }
        return LazyUOpCache.intern(Kind.BUFFER, new LazyUOp[0], null, tensor, null);
    }

    static LazyUOp unary(Kind kind, LazyUOp input, int[] arg) {
        validateUnaryKind(kind);
        return LazyUOpCache.intern(kind, new LazyUOp[] {input}, arg, null, null);
    }

    static LazyUOp binary(Kind kind, LazyUOp left, LazyUOp right) {
        return binary(kind, left, right, null);
    }

    static LazyUOp binary(Kind kind, LazyUOp left, LazyUOp right, int[] arg) {
        validateBinaryKind(kind);
        return LazyUOpCache.intern(kind, new LazyUOp[] {left, right}, arg, null, null);
    }

    static LazyUOp ternary(Kind kind, LazyUOp first, LazyUOp second, LazyUOp third) {
        if (kind != Kind.WHERE && kind != Kind.CONV2D_INPUT_GRAD && kind != Kind.CONV2D_WEIGHT_GRAD
                && kind != Kind.CONV_TRANSPOSE2D_INPUT_GRAD && kind != Kind.CONV_TRANSPOSE2D_WEIGHT_GRAD) {
            throw new IllegalArgumentException("unsupported ternary kind " + kind);
        }
        return LazyUOpCache.intern(kind, new LazyUOp[] {first, second, third}, null, null, null);
    }

    static LazyUOp ternary(Kind kind, LazyUOp first, LazyUOp second, LazyUOp third, int[] arg) {
        if (kind != Kind.CONV2D_INPUT_GRAD && kind != Kind.CONV2D_WEIGHT_GRAD
                && kind != Kind.CONV_TRANSPOSE2D_INPUT_GRAD && kind != Kind.CONV_TRANSPOSE2D_WEIGHT_GRAD) {
            throw new IllegalArgumentException("unsupported ternary kind with arg " + kind);
        }
        return LazyUOpCache.intern(kind, new LazyUOp[] {first, second, third}, arg, null, null);
    }

    static LazyUOp multi(Kind kind, LazyUOp[] src, int[] arg) {
        validateMultiKind(kind, src.length);
        return LazyUOpCache.intern(kind, src, arg, null, null);
    }

    public Kind op() {
        return op;
    }

    public int srcCount() {
        return src.length;
    }

    public LazyUOp src(int index) {
        return src[index];
    }

    public LazyUOp[] src() {
        return src.clone();
    }

    public int[] arg() {
        return arg == null ? null : arg.clone();
    }

    public Tensor buffer() {
        return buffer;
    }

    public LazyGgufSlice ggufSlice() {
        return ggufSlice;
    }

    /** @deprecated use {@link #ggufSlice()} */
    public LazyGgufSlice quantSlice() {
        return ggufSlice;
    }

    public boolean isGgufLeaf() {
        return op == Kind.DEQUANT_Q4_0 || op == Kind.MMAP_F32;
    }

    public boolean isMovementOnly() {
        if (op == Kind.BUFFER) {
            return true;
        }
        if (op == Kind.RESHAPE || op == Kind.PERMUTE || op == Kind.EXPAND) {
            return src[0].isMovementOnly();
        }
        return false;
    }

    public LazyShape movementShape() {
        if (!isMovementOnly()) {
            throw new IllegalStateException("movementShape requires movement-only graph, got " + op);
        }
        return movementShapeNode(this);
    }

    private static LazyShape movementShapeNode(LazyUOp node) {
        switch (node.op) {
            case BUFFER:
                return LazyShape.leaf(node.buffer.shape().dims());
            case RESHAPE:
                return movementShapeNode(node.src[0]).reshape(node.arg);
            case PERMUTE:
                return movementShapeNode(node.src[0]).permute(node.arg);
            case EXPAND:
                return movementShapeNode(node.src[0]).expand(node.arg);
            default:
                throw new IllegalStateException("unexpected op " + node.op);
        }
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof LazyUOp lazyUOp)) {
            return false;
        }
        return op == lazyUOp.op && Arrays.equals(src, lazyUOp.src) && Arrays.equals(arg, lazyUOp.arg)
                && buffer == lazyUOp.buffer && ggufSlice == lazyUOp.ggufSlice;
    }

    @Override
    public int hashCode() {
        return Objects.hash(op, Arrays.hashCode(src), Arrays.hashCode(arg), buffer, ggufSlice);
    }

    @Override
    public String toString() {
        return "LazyUOp{" + op + " src=" + src.length + " arg=" + Arrays.toString(arg) + "}";
    }

    private static void validateUnaryKind(Kind kind) {
        if (kind != Kind.RESHAPE && kind != Kind.PERMUTE && kind != Kind.EXPAND
                && kind != Kind.PAD && kind != Kind.SHRINK && kind != Kind.CONTIGUOUS && kind != Kind.CAST
                && kind != Kind.NEG && kind != Kind.RECIP && kind != Kind.POW
                && kind != Kind.LOG2 && kind != Kind.EXP2 && kind != Kind.EXP && kind != Kind.SQRT
                && kind != Kind.RELU && kind != Kind.RELU_MASK
                && kind != Kind.SUM && kind != Kind.SUM_AXIS && kind != Kind.MAX_AXIS && kind != Kind.MEAN
                && kind != Kind.IM2COL && kind != Kind.IM2COL_INPUT_GRAD && kind != Kind.POOL2D
                && kind != Kind.MAX_UNPOOL2D_VALUE_GRAD) {
            throw new IllegalArgumentException("unsupported unary kind " + kind);
        }
    }

    private static void validateBinaryKind(Kind kind) {
        if (kind != Kind.ADD && kind != Kind.MUL && kind != Kind.SUB && kind != Kind.DIV
                && kind != Kind.MAX && kind != Kind.GT_MASK && kind != Kind.EQ_MASK
                && kind != Kind.CONV2D && kind != Kind.CONV_TRANSPOSE2D) {
            throw new IllegalArgumentException("unsupported binary kind " + kind);
        }
    }

    private static void validateMultiKind(Kind kind, int srcLen) {
        if (kind == Kind.MAX_UNPOOL2D && srcLen == 2) {
            return;
        }
        if (kind == Kind.MAX_UNPOOL2D_VALUE_GRAD && srcLen == 2) {
            return;
        }
        throw new IllegalArgumentException("unsupported multi kind " + kind + " src=" + srcLen);
    }
}
