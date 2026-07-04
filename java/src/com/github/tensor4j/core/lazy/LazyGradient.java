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
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Symbolic backward on {@link LazyUOp} DAGs (tinygrad {@code compute_gradient} / {@code pm_gradient} subset).
 */
final class LazyGradient {

    private LazyGradient() {
    }

    static Map<LazyUOp, LazyUOp> compute(LazyUOp root, LazyUOp rootGrad, Set<LazyUOp> targets) {
        Map<LazyUOp, LazyUOp> grads = new HashMap<>();
        grads.put(root, rootGrad);
        List<LazyUOp> order = LazyGraph.deepwalk(root, targets);
        for (int index = order.size() - 1; index >= 0; index--) {
            LazyUOp node = order.get(index);
            LazyUOp ctx = grads.get(node);
            if (ctx == null) {
                continue;
            }
            applyRule(node, ctx, grads);
        }
        Map<LazyUOp, LazyUOp> result = new HashMap<>();
        for (LazyUOp target : targets) {
            LazyUOp grad = grads.get(target);
            if (grad != null) {
                result.put(target, grad);
            }
        }
        return result;
    }

    private static void applyRule(LazyUOp node, LazyUOp ctx, Map<LazyUOp, LazyUOp> grads) {
        switch (node.op()) {
            case BUFFER, RELU_MASK, GT_MASK, EQ_MASK, CONV2D_INPUT_GRAD, CONV2D_WEIGHT_GRAD, IM2COL_INPUT_GRAD,
                    CONV_TRANSPOSE2D_INPUT_GRAD, CONV_TRANSPOSE2D_WEIGHT_GRAD, MAX_UNPOOL2D_VALUE_GRAD -> {
            }
            case CAST, CONTIGUOUS -> accumulate(grads, node.src(0), ctx);
            case ADD -> {
                accumulate(grads, node.src(0), ctx);
                accumulate(grads, node.src(1), ctx);
            }
            case MUL -> {
                accumulate(grads, node.src(0), LazyUOp.binary(LazyUOp.Kind.MUL, node.src(1), ctx));
                accumulate(grads, node.src(1), LazyUOp.binary(LazyUOp.Kind.MUL, node.src(0), ctx));
            }
            case SUB -> {
                accumulate(grads, node.src(0), ctx);
                accumulate(grads, node.src(1), LazyUOp.unary(LazyUOp.Kind.NEG, ctx, null));
            }
            case DIV -> {
                LazyUOp recipRight = LazyUOp.unary(LazyUOp.Kind.RECIP, node.src(1), null);
                accumulate(grads, node.src(0), LazyUOp.binary(LazyUOp.Kind.MUL, ctx, recipRight));
                LazyUOp recipSq = LazyUOp.binary(LazyUOp.Kind.MUL, recipRight, recipRight);
                LazyUOp gradRight = LazyUOp.unary(LazyUOp.Kind.NEG,
                        LazyUOp.binary(LazyUOp.Kind.MUL, LazyUOp.binary(LazyUOp.Kind.MUL, ctx, node.src(0)), recipSq),
                        null);
                accumulate(grads, node.src(1), gradRight);
            }
            case MAX -> {
                LazyUOp gt = LazyUOp.binary(LazyUOp.Kind.GT_MASK, node.src(0), node.src(1));
                LazyUOp eq = LazyUOp.binary(LazyUOp.Kind.EQ_MASK, node.src(0), node.src(1));
                LazyUOp halfEq = LazyUOp.binary(LazyUOp.Kind.MUL, eq, scalarBuffer(0.5f));
                accumulate(grads, node.src(0), LazyUOp.binary(LazyUOp.Kind.MUL, ctx,
                        LazyUOp.binary(LazyUOp.Kind.ADD, gt, halfEq)));
                LazyUOp one = scalarBuffer(1f);
                LazyUOp gradRightMask = LazyUOp.binary(LazyUOp.Kind.SUB, one,
                        LazyUOp.binary(LazyUOp.Kind.ADD, gt, halfEq));
                accumulate(grads, node.src(1), LazyUOp.binary(LazyUOp.Kind.MUL, ctx, gradRightMask));
            }
            case WHERE -> {
                LazyUOp cond = node.src(0);
                LazyUOp one = scalarBuffer(1f);
                LazyUOp invCond = LazyUOp.binary(LazyUOp.Kind.SUB, one, cond);
                accumulate(grads, node.src(1), LazyUOp.binary(LazyUOp.Kind.MUL, ctx, cond));
                accumulate(grads, node.src(2), LazyUOp.binary(LazyUOp.Kind.MUL, ctx, invCond));
            }
            case RECIP -> accumulate(grads, node.src(0), LazyUOp.unary(LazyUOp.Kind.NEG,
                    LazyUOp.binary(LazyUOp.Kind.MUL, LazyUOp.binary(LazyUOp.Kind.MUL, ctx, node), node), null));
            case NEG -> accumulate(grads, node.src(0), LazyUOp.unary(LazyUOp.Kind.NEG, ctx, null));
            case POW -> {
                float exponent = LazyMath.floatArg(node.arg()[0]);
                LazyUOp base = node.src(0);
                LazyUOp powMinusOne = LazyUOp.unary(LazyUOp.Kind.POW, base,
                        new int[] {LazyMath.floatArg(exponent - 1f)});
                accumulate(grads, base, LazyUOp.binary(LazyUOp.Kind.MUL,
                        LazyUOp.binary(LazyUOp.Kind.MUL, ctx, scalarBuffer(exponent)), powMinusOne));
            }
            case LOG2 -> {
                LazyUOp denom = LazyUOp.binary(LazyUOp.Kind.MUL, node.src(0), scalarBuffer(LazyMath.LN2));
                accumulate(grads, node.src(0), LazyUOp.binary(LazyUOp.Kind.MUL, ctx,
                        LazyUOp.unary(LazyUOp.Kind.RECIP, denom, null)));
            }
            case EXP2 -> accumulate(grads, node.src(0), LazyUOp.binary(LazyUOp.Kind.MUL,
                    LazyUOp.binary(LazyUOp.Kind.MUL, ctx, node), scalarBuffer(LazyMath.LN2)));
            case EXP -> accumulate(grads, node.src(0), LazyUOp.binary(LazyUOp.Kind.MUL, ctx, node));
            case SQRT -> accumulate(grads, node.src(0), LazyUOp.binary(LazyUOp.Kind.MUL,
                    LazyUOp.binary(LazyUOp.Kind.MUL, ctx, scalarBuffer(0.5f)),
                    LazyUOp.unary(LazyUOp.Kind.RECIP, node, null)));
            case RELU -> accumulate(grads, node.src(0),
                    LazyUOp.binary(LazyUOp.Kind.MUL, ctx, LazyUOp.unary(LazyUOp.Kind.RELU_MASK, node.src(0), null)));
            case SUM -> accumulate(grads, node.src(0), broadcastScalarTo(ctx, LazyUOpShapes.infer(node.src(0))));
            case MEAN -> {
                int count = Strides.numel(LazyUOpShapes.infer(node.src(0)));
                LazyUOp scaled = LazyUOp.binary(LazyUOp.Kind.MUL, ctx, scalarBuffer(1f / count));
                accumulate(grads, node.src(0), broadcastScalarTo(scaled, LazyUOpShapes.infer(node.src(0))));
            }
            case SUM_AXIS -> accumulate(grads, node.src(0), sumAxisGrad(ctx, node));
            case MAX_AXIS -> {
                LazyUOp input = node.src(0);
                int axis = node.arg()[0];
                LazyUOp eq = LazyUOp.binary(LazyUOp.Kind.EQ_MASK, input, node);
                LazyUOp sumEq = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, eq, LazySoftmax.axisArg(axis, true));
                LazyUOp mask = LazyUOp.binary(LazyUOp.Kind.DIV, eq, sumEq);
                accumulate(grads, input, LazyUOp.binary(LazyUOp.Kind.MUL, ctx, mask));
            }
            case CONV2D -> {
                LazyUOp input = node.src(0);
                LazyUOp weight = node.src(1);
                accumulate(grads, input, LazyUOp.ternary(LazyUOp.Kind.CONV2D_INPUT_GRAD,
                        ctx, weight, input, node.arg()));
                accumulate(grads, weight, LazyUOp.ternary(LazyUOp.Kind.CONV2D_WEIGHT_GRAD,
                        ctx, input, weight, node.arg()));
            }
            case CONV_TRANSPOSE2D -> {
                LazyUOp input = node.src(0);
                LazyUOp weight = node.src(1);
                accumulate(grads, input, LazyUOp.ternary(LazyUOp.Kind.CONV_TRANSPOSE2D_INPUT_GRAD,
                        ctx, weight, input, node.arg()));
                accumulate(grads, weight, LazyUOp.ternary(LazyUOp.Kind.CONV_TRANSPOSE2D_WEIGHT_GRAD,
                        ctx, input, weight, node.arg()));
            }
            case RESHAPE -> {
                int[] outShape = LazyUOpShapes.infer(node);
                int[] srcShape = LazyUOpShapes.infer(node.src(0));
                LazyUOp aligned = reduceBroadcastAxes(ctx, outShape);
                accumulate(grads, node.src(0), LazyUOp.unary(LazyUOp.Kind.RESHAPE, aligned, srcShape));
            }
            case PERMUTE -> {
                LazyUOp aligned = reduceBroadcastAxes(ctx, LazyUOpShapes.infer(node));
                accumulate(grads, node.src(0),
                        LazyUOp.unary(LazyUOp.Kind.PERMUTE, aligned, Strides.inversePermute(node.arg())));
            }
            case EXPAND -> accumulate(grads, node.src(0), expandGrad(ctx, node));
            case PAD -> {
                int[] srcShape = LazyUOpShapes.infer(node.src(0));
                int[] shrinkArg = LazyPadShrink.padBackwardShrinkArg(srcShape, node.arg());
                accumulate(grads, node.src(0), LazyUOp.unary(LazyUOp.Kind.SHRINK, ctx, shrinkArg));
            }
            case SHRINK -> {
                int[] srcShape = LazyUOpShapes.infer(node.src(0));
                int[] padArg = LazyPadShrink.shrinkBackwardPadArg(srcShape, node.arg());
                accumulate(grads, node.src(0), LazyUOp.unary(LazyUOp.Kind.PAD, ctx, padArg));
            }
            case IM2COL -> {
                int[] paddedShape = LazyUOpShapes.infer(node.src(0));
                accumulate(grads, node.src(0), LazyUOp.unary(LazyUOp.Kind.IM2COL_INPUT_GRAD, ctx,
                        LazyConv2d.im2colGradArg(node.arg(), paddedShape)));
            }
            case MAX_UNPOOL2D -> {
                int[] valueShape = LazyUOpShapes.infer(node.src(0));
                accumulate(grads, node.src(0), LazyUOp.multi(LazyUOp.Kind.MAX_UNPOOL2D_VALUE_GRAD,
                        new LazyUOp[] {ctx, node.src(1)}, LazyMaxUnpool2d.gradArg(node.arg(), valueShape)));
            }
            default -> throw new IllegalStateException("no gradient rule for " + node.op());
        }
    }

    private static void accumulate(Map<LazyUOp, LazyUOp> grads, LazyUOp target, LazyUOp contribution) {
        LazyUOp reduced = reduceBroadcastAxes(contribution, LazyUOpShapes.infer(target));
        LazyUOp existing = grads.get(target);
        if (existing == null) {
            grads.put(target, reduced);
        } else {
            grads.put(target, LazyUOp.binary(LazyUOp.Kind.ADD, existing, reduced));
        }
    }

    private static LazyUOp reduceBroadcastAxes(LazyUOp grad, int[] targetShape) {
        int[] current = LazyUOpShapes.infer(grad);
        if (java.util.Arrays.equals(current, targetShape)) {
            return grad;
        }
        if (Strides.numel(current) == Strides.numel(targetShape)) {
            return LazyUOp.unary(LazyUOp.Kind.RESHAPE, grad, targetShape);
        }
        LazyUOp out = grad;
        for (int guard = 0; guard < 16; guard++) {
            current = LazyUOpShapes.infer(out);
            if (java.util.Arrays.equals(current, targetShape)) {
                return out;
            }
            if (Strides.numel(current) == Strides.numel(targetShape)) {
                return LazyUOp.unary(LazyUOp.Kind.RESHAPE, out, targetShape);
            }
            int rank = Math.max(current.length, targetShape.length);
            int[] cur = padShape(current, rank);
            int[] tgt = padShape(targetShape, rank);
            int sumAxis = -1;
            for (int axis = rank - 1; axis >= 0; axis--) {
                if (tgt[axis] == 1 && cur[axis] > 1) {
                    sumAxis = axis;
                    break;
                }
            }
            if (sumAxis < 0) {
                break;
            }
            out = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, out, new int[] {sumAxis});
        }
        current = LazyUOpShapes.infer(out);
        if (!java.util.Arrays.equals(current, targetShape)) {
            if (Strides.numel(current) == Strides.numel(targetShape)) {
                out = LazyUOp.unary(LazyUOp.Kind.RESHAPE, out, targetShape);
            } else {
                throw new IllegalStateException("cannot reduce grad shape "
                        + java.util.Arrays.toString(current) + " to " + java.util.Arrays.toString(targetShape));
            }
        }
        return out;
    }

    private static int[] padShape(int[] shape, int rank) {
        if (shape.length >= rank) {
            return shape.clone();
        }
        return Strides.alignLeft(shape, rank);
    }

    private static LazyUOp broadcastScalarTo(LazyUOp ctx, int[] targetShape) {
        int[] ctxShape = LazyUOpShapes.infer(ctx);
        if (Strides.numel(ctxShape) == 1 && Strides.numel(targetShape) > 1) {
            LazyUOp expanded = LazyUOp.unary(LazyUOp.Kind.RESHAPE, ctx, leftOnes(targetShape.length));
            return LazyUOp.unary(LazyUOp.Kind.EXPAND, expanded, targetShape);
        }
        if (!java.util.Arrays.equals(ctxShape, targetShape)) {
            return LazyUOp.unary(LazyUOp.Kind.EXPAND, LazyUOp.unary(LazyUOp.Kind.RESHAPE, ctx, targetShape), targetShape);
        }
        return ctx;
    }

    private static int[] leftOnes(int rank) {
        int[] shape = new int[rank];
        java.util.Arrays.fill(shape, 1);
        return shape;
    }

    private static LazyUOp scalarBuffer(float value) {
        return LazyUOp.buffer(Tensor.of(value));
    }

    private static LazyUOp sumAxisGrad(LazyUOp ctx, LazyUOp sumNode) {
        LazyUOp input = sumNode.src(0);
        int axis = sumNode.arg()[0];
        boolean keepdim = sumNode.arg().length > 1 && sumNode.arg()[1] == 1;
        int[] inputShape = LazyUOpShapes.infer(input);
        if (keepdim) {
            return LazyUOp.unary(LazyUOp.Kind.EXPAND, ctx, inputShape);
        }
        int[] insertShape = inputShape.clone();
        insertShape[axis] = 1;
        LazyUOp reshaped = LazyUOp.unary(LazyUOp.Kind.RESHAPE, ctx, insertShape);
        return LazyUOp.unary(LazyUOp.Kind.EXPAND, reshaped, inputShape);
    }

    private static LazyUOp expandGrad(LazyUOp ctx, LazyUOp expandNode) {
        LazyUOp src = expandNode.src(0);
        int[] srcShape = LazyUOpShapes.infer(src);
        int[] outShape = LazyUOpShapes.infer(expandNode);
        LazyUOp grad = ctx;
        for (int axis = outShape.length - 1; axis >= 0; axis--) {
            if (srcShape[axis] != outShape[axis] && srcShape[axis] == 1) {
                grad = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, grad, new int[] {axis});
            }
        }
        if (!java.util.Arrays.equals(LazyUOpShapes.infer(grad), srcShape)) {
            grad = LazyUOp.unary(LazyUOp.Kind.RESHAPE, grad, srcShape);
        }
        return grad;
    }

    static Set<LazyUOp> gradTargets(LazyUOp root) {
        Set<LazyUOp> targets = new LinkedHashSet<>();
        for (LazyUOp node : LazyGraph.toposort(root)) {
            if (node.op() == LazyUOp.Kind.BUFFER && node.buffer().requiresGrad()) {
                targets.add(node);
            }
        }
        return targets;
    }
}
