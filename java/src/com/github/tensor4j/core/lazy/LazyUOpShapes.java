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

import com.github.tensor4j.core.AdaptivePool2dMath;
import com.github.tensor4j.core.BatchNorm2dMath;
import com.github.tensor4j.core.Conv2dMath;
import com.github.tensor4j.core.ConvTranspose2dMath;
import com.github.tensor4j.core.Pool2dMath;
import java.util.HashMap;
import java.util.Map;

/** Shape inference for {@link LazyUOp} DAGs (tinygrad {@code UOp._shape}). */
final class LazyUOpShapes {

    private LazyUOpShapes() {
    }

    static int[] infer(LazyUOp root) {
        Map<LazyUOp, int[]> memo = new HashMap<>();
        return inferNode(root, memo).clone();
    }

    static void validateOrThrow(LazyUOp root) {
        infer(root);
    }

    private static int[] inferNode(LazyUOp node, Map<LazyUOp, int[]> memo) {
        int[] cached = memo.get(node);
        if (cached != null) {
            return cached;
        }
        int[] shape = computeShape(node, memo);
        memo.put(node, shape);
        return shape;
    }

    private static int[] computeShape(LazyUOp node, Map<LazyUOp, int[]> memo) {
        switch (node.op()) {
            case BUFFER:
                return node.buffer().shape().dims();
            case RESHAPE:
                return LazyShape.leaf(inferNode(node.src(0), memo)).reshape(node.arg()).shape();
            case PERMUTE:
                return LazyShape.leaf(inferNode(node.src(0), memo)).permute(node.arg()).shape();
            case EXPAND:
                return LazyShape.leaf(inferNode(node.src(0), memo)).expand(node.arg()).shape();
            case PAD:
                return LazyPadShrink.paddedShape(inferNode(node.src(0), memo), node.arg());
            case SHRINK:
                return LazyPadShrink.shrinkShape(inferNode(node.src(0), memo), node.arg());
            case CONTIGUOUS:
            case CAST:
            case NEG:
            case RECIP:
            case POW:
            case LOG2:
            case EXP2:
            case EXP:
            case SQRT:
            case RELU:
            case RELU_MASK:
                return inferNode(node.src(0), memo).clone();
            case ADD:
            case MUL:
            case SUB:
            case DIV:
            case MAX:
            case GT_MASK:
            case EQ_MASK:
                return LazyShape.broadcastShape(inferNode(node.src(0), memo), inferNode(node.src(1), memo));
            case WHERE:
                return LazyShape.broadcastShape(
                        LazyShape.broadcastShape(inferNode(node.src(0), memo), inferNode(node.src(1), memo)),
                        inferNode(node.src(2), memo));
            case SUM:
            case MEAN:
                return new int[] {1};
            case SUM_AXIS:
                return reduceAxisShape(inferNode(node.src(0), memo), node.arg());
            case MAX_AXIS:
                return reduceAxisShape(inferNode(node.src(0), memo), node.arg());
            case CONV2D:
                return Conv2dMath.outputShape(
                        inferNode(node.src(0), memo),
                        inferNode(node.src(1), memo),
                        node.arg());
            case CONV2D_INPUT_GRAD:
                return inferNode(node.src(2), memo).clone();
            case CONV2D_WEIGHT_GRAD:
                return inferNode(node.src(2), memo).clone();
            case IM2COL:
                return LazyConv2d.im2colWindowShape(inferNode(node.src(0), memo), node.arg());
            case IM2COL_INPUT_GRAD:
                return LazyConv2d.paddedShapeFromGradArg(node.arg());
            case POOL2D:
                return Pool2dMath.outputShape(inferNode(node.src(0), memo), node.arg());
            case CONV_TRANSPOSE2D:
                return ConvTranspose2dMath.outputShape(
                        inferNode(node.src(0), memo),
                        inferNode(node.src(1), memo),
                        node.arg());
            case CONV_TRANSPOSE2D_INPUT_GRAD:
                return inferNode(node.src(2), memo).clone();
            case CONV_TRANSPOSE2D_WEIGHT_GRAD:
                return inferNode(node.src(2), memo).clone();
            case MAX_UNPOOL2D:
                return LazyMaxUnpool2d.outputShapeFromPacked(node.arg());
            case MAX_UNPOOL2D_VALUE_GRAD:
                return LazyMaxUnpool2d.valueShapeFromGradPacked(node.arg());
            default:
                throw new IllegalStateException("unhandled op " + node.op());
        }
    }

    private static int[] reduceAxisShape(int[] inputShape, int[] arg) {
        int axis = arg[0];
        boolean keepdim = arg.length > 1 && arg[1] == 1;
        if (keepdim) {
            int[] out = inputShape.clone();
            out[axis] = 1;
            return out;
        }
        return LazyShape.leaf(inputShape).reduceAxis(axis).shape();
    }
}
