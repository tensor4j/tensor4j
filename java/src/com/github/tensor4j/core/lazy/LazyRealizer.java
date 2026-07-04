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
import com.github.tensor4j.core.ConvIm2Col;
import com.github.tensor4j.core.ConvTranspose2dMath;
import com.github.tensor4j.core.MaxUnpool2dMath;
import com.github.tensor4j.core.Pool2dMath;
import com.github.tensor4j.core.SoftmaxMath;
import com.github.tensor4j.core.Strides;
import com.github.tensor4j.core.Tensor;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Materializes {@link LazyUOp} DAGs into eager {@link Tensor} values.
 * Memoizes per-node results so shared subgraphs compute once (tinygrad DAG semantics).
 */
final class LazyRealizer {

    private LazyRealizer() {
    }

    static Tensor realize(LazyUOp root) {
        return LazySchedule.build(root).execute();
    }

    static Tensor realize(LazyUOp root, boolean enableGrad) {
        return LazySchedule.build(root).execute();
    }

    static Tensor realize(LazyUOp root, boolean enableGrad, Map<LazyUOp, Tensor> memo) {
        return realizeNode(root, enableGrad, memo);
    }

    private static Tensor realizeNode(LazyUOp node, boolean enableGrad, Map<LazyUOp, Tensor> memo) {
        Tensor cached = memo.get(node);
        if (cached != null) {
            return cached;
        }
        Tensor value = switch (node.op()) {
            case BUFFER -> node.buffer();
            case RESHAPE -> realizeNode(node.src(0), enableGrad, memo).reshape(node.arg());
            case PERMUTE -> realizeNode(node.src(0), enableGrad, memo).permute(node.arg());
            case EXPAND -> realizeNode(node.src(0), enableGrad, memo).expand(node.arg());
            case ADD -> broadcastBinary(realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo), BinaryKind.ADD, enableGrad);
            case MUL -> broadcastBinary(realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo), BinaryKind.MUL, enableGrad);
            case SUB -> broadcastBinary(realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo), BinaryKind.SUB, enableGrad);
            case DIV -> broadcastBinary(realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo), BinaryKind.DIV, enableGrad);
            case NEG -> unary(realizeNode(node.src(0), enableGrad, memo), UnaryKind.NEG, enableGrad);
            case RECIP -> recip(realizeNode(node.src(0), enableGrad, memo));
            case RELU -> unary(realizeNode(node.src(0), enableGrad, memo), UnaryKind.RELU, enableGrad);
            case RELU_MASK -> reluMask(realizeNode(node.src(0), false, memo));
            case SUM -> unary(realizeNode(node.src(0), enableGrad, memo), UnaryKind.SUM, enableGrad);
            case MEAN -> mean(realizeNode(node.src(0), enableGrad, memo));
            case SUM_AXIS -> sumAxis(realizeNode(node.src(0), enableGrad, memo), node.arg());
            case MAX_AXIS -> maxAxis(realizeNode(node.src(0), enableGrad, memo), node.arg());
            case EXP -> exp(realizeNode(node.src(0), enableGrad, memo));
            case CONV2D -> Conv2dMath.forward(
                    realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    node.arg());
            case CONV2D_INPUT_GRAD -> Conv2dMath.gradInput(
                    realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    node.arg(),
                    LazyUOpShapes.infer(node.src(2)));
            case CONV2D_WEIGHT_GRAD -> Conv2dMath.gradWeight(
                    realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    node.arg(),
                    LazyUOpShapes.infer(node.src(2)));
            case PAD -> LazyPadShrink.applyPad(realizeNode(node.src(0), enableGrad, memo), node.arg());
            case SHRINK -> LazyPadShrink.applyShrink(realizeNode(node.src(0), enableGrad, memo), node.arg());
            case CONTIGUOUS -> LazyMath.contiguousCopy(realizeNode(node.src(0), enableGrad, memo));
            case CAST -> realizeNode(node.src(0), enableGrad, memo);
            case POW -> pow(realizeNode(node.src(0), enableGrad, memo), LazyMath.floatArg(node.arg()[0]));
            case LOG2 -> log2(realizeNode(node.src(0), enableGrad, memo));
            case EXP2 -> exp2(realizeNode(node.src(0), enableGrad, memo));
            case SQRT -> sqrt(realizeNode(node.src(0), enableGrad, memo));
            case MAX -> broadcastBinary(realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo), BinaryKind.MAX, enableGrad);
            case GT_MASK -> gtMask(realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo));
            case EQ_MASK -> eqMask(realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo));
            case WHERE -> where(realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    realizeNode(node.src(2), enableGrad, memo));
            case IM2COL -> {
                int[] im2colArg = node.arg();
                Tensor padded = realizeNode(node.src(0), enableGrad, memo);
                int[] winShape = LazyConv2d.im2colWindowShape(padded.shape().dims(), im2colArg);
                yield Tensor.of(LazyConv2d.realizeIm2Col(padded, im2colArg), winShape);
            }
            case IM2COL_INPUT_GRAD -> {
                int[] paddedShape = LazyConv2d.paddedShapeFromGradArg(node.arg());
                int[] im2colArg = LazyConv2d.im2colArgFromGradArg(node.arg());
                yield ConvIm2Col.scatterWindowsGrad(realizeNode(node.src(0), enableGrad, memo), paddedShape,
                        im2colArg);
            }
            case POOL2D -> Pool2dMath.forward(realizeNode(node.src(0), enableGrad, memo), node.arg());
            case CONV_TRANSPOSE2D -> ConvTranspose2dMath.forward(
                    realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    node.arg());
            case CONV_TRANSPOSE2D_INPUT_GRAD -> ConvTranspose2dMath.gradInput(
                    realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    node.arg(),
                    LazyUOpShapes.infer(node.src(2)));
            case CONV_TRANSPOSE2D_WEIGHT_GRAD -> ConvTranspose2dMath.gradWeight(
                    realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    node.arg(),
                    LazyUOpShapes.infer(node.src(2)));
            case MAX_UNPOOL2D -> MaxUnpool2dMath.forward(
                    realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    LazyMaxUnpool2d.poolArgFromPacked(node.arg()),
                    LazyMaxUnpool2d.outputShapeFromPacked(node.arg()));
            case MAX_UNPOOL2D_VALUE_GRAD -> MaxUnpool2dMath.gradValues(
                    realizeNode(node.src(0), enableGrad, memo),
                    realizeNode(node.src(1), enableGrad, memo),
                    LazyMaxUnpool2d.poolArgFromPacked(node.arg()),
                    LazyMaxUnpool2d.valueShapeFromGradPacked(node.arg()));
            case DEQUANT_Q4_0 -> {
                LazyGgufSlice slice = node.ggufSlice();
                float[] data = LazyQuantMath.dequantizeQ4_0(
                        slice.buffer(), slice.offset(), slice.numElements());
                yield Tensor.of(data, slice.floatShape());
            }
            case MMAP_F32 -> {
                LazyGgufSlice slice = node.ggufSlice();
                float[] data = LazyQuantMath.readF32(slice.buffer(), slice.offset(), slice.numElements());
                yield Tensor.of(data, slice.floatShape());
            }
            default -> throw new IllegalStateException("unsupported op " + node.op());
        };
        memo.put(node, value);
        return value;
    }

    private enum BinaryKind {
        ADD,
        MUL,
        SUB,
        DIV,
        MAX
    }

    private enum UnaryKind {
        NEG,
        RELU,
        SUM
    }

    private static Tensor recip(Tensor input) {
        int n = input.numel();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = 1f / input.getFlat(i);
        }
        return Tensor.of(out, input.shape().dims());
    }

    private static Tensor mean(Tensor input) {
        float total = 0f;
        int n = input.numel();
        for (int i = 0; i < n; i++) {
            total += input.getFlat(i);
        }
        return Tensor.of(total / n);
    }

    private static Tensor exp(Tensor input) {
        int n = input.numel();
        return Tensor.of(SoftmaxMath.expInto(input.toFlatArray(), n), input.shape().dims());
    }

    private static Tensor sumAxis(Tensor input, int[] arg) {
        int axis = arg[0];
        boolean keepdim = arg.length > 1 && arg[1] == 1;
        return SoftmaxMath.sumAxis(input, axis, keepdim);
    }

    private static Tensor maxAxis(Tensor input, int[] arg) {
        int axis = arg[0];
        boolean keepdim = arg.length > 1 && arg[1] == 1;
        return SoftmaxMath.maxAxis(input, axis, keepdim);
    }

    private static Tensor reluMask(Tensor input) {
        int n = input.numel();
        float[] mask = new float[n];
        for (int i = 0; i < n; i++) {
            mask[i] = input.getFlat(i) > 0f ? 1f : 0f;
        }
        return Tensor.of(mask, input.shape().dims());
    }

    private static Tensor pow(Tensor input, float exponent) {
        int n = input.numel();
        return Tensor.of(LazyMath.powInto(input.toFlatArray(), exponent, n), input.shape().dims());
    }

    private static Tensor log2(Tensor input) {
        int n = input.numel();
        return Tensor.of(LazyMath.log2Into(input.toFlatArray(), n), input.shape().dims());
    }

    private static Tensor exp2(Tensor input) {
        int n = input.numel();
        return Tensor.of(LazyMath.exp2Into(input.toFlatArray(), n), input.shape().dims());
    }

    private static Tensor sqrt(Tensor input) {
        int n = input.numel();
        return Tensor.of(LazyMath.sqrtInto(input.toFlatArray(), n), input.shape().dims());
    }

    private static Tensor gtMask(Tensor left, Tensor right) {
        int[] outShape = LazyShape.broadcastShape(left.shape().dims(), right.shape().dims());
        int n = Strides.numel(outShape);
        float[] result = LazyMath.gtMaskInto(
                expandToShape(left, outShape).toFlatArray(),
                expandToShape(right, outShape).toFlatArray(), n);
        return Tensor.of(result, outShape);
    }

    private static Tensor eqMask(Tensor left, Tensor right) {
        int[] outShape = LazyShape.broadcastShape(left.shape().dims(), right.shape().dims());
        int n = Strides.numel(outShape);
        float[] result = LazyMath.eqMaskInto(
                expandToShape(left, outShape).toFlatArray(),
                expandToShape(right, outShape).toFlatArray(), n);
        return Tensor.of(result, outShape);
    }

    private static Tensor where(Tensor cond, Tensor ifTrue, Tensor ifFalse) {
        int[] outShape = LazyShape.broadcastShape(
                LazyShape.broadcastShape(cond.shape().dims(), ifTrue.shape().dims()),
                ifFalse.shape().dims());
        int n = Strides.numel(outShape);
        float[] result = LazyMath.whereInto(
                expandToShape(cond, outShape).toFlatArray(),
                expandToShape(ifTrue, outShape).toFlatArray(),
                expandToShape(ifFalse, outShape).toFlatArray(), n);
        return Tensor.of(result, outShape);
    }

    private static Tensor unary(Tensor input, UnaryKind op, boolean enableGrad) {
        if (!enableGrad) {
            return switch (op) {
                case NEG -> input.neg().withGrad(false);
                case RELU -> input.relu().withGrad(false);
                case SUM -> input.sum().withGrad(false);
            };
        }
        return switch (op) {
            case NEG -> input.neg();
            case RELU -> input.relu();
            case SUM -> input.sum();
        };
    }

    private static Tensor broadcastBinary(Tensor left, Tensor right, BinaryKind op, boolean enableGrad) {
        int[] outShape = LazyShape.broadcastShape(left.shape().dims(), right.shape().dims());
        Tensor leftExpanded = expandToShape(left, outShape);
        Tensor rightExpanded = expandToShape(right, outShape);
        if (enableGrad) {
            return switch (op) {
                case ADD -> leftExpanded.add(rightExpanded);
                case MUL -> leftExpanded.mul(rightExpanded);
                case SUB -> leftExpanded.sub(rightExpanded);
                case DIV -> leftExpanded.div(rightExpanded);
                case MAX -> Tensor.of(LazyMath.maxInto(
                        leftExpanded.toFlatArray(), rightExpanded.toFlatArray(), Strides.numel(outShape)), outShape);
            };
        }
        int n = Strides.numel(outShape);
        float[] result = new float[n];
        for (int i = 0; i < n; i++) {
            float leftValue = leftExpanded.getFlat(i);
            float rightValue = rightExpanded.getFlat(i);
            result[i] = switch (op) {
                case ADD -> leftValue + rightValue;
                case MUL -> leftValue * rightValue;
                case SUB -> leftValue - rightValue;
                case DIV -> leftValue / rightValue;
                case MAX -> Math.max(leftValue, rightValue);
            };
        }
        return Tensor.of(result, outShape);
    }

    static Tensor expandToShape(Tensor tensor, int[] targetShape) {
        int[] shape = tensor.shape().dims();
        if (Arrays.equals(shape, targetShape)) {
            return tensor;
        }
        if (shape.length < targetShape.length) {
            tensor = tensor.reshape(Strides.alignLeft(shape, targetShape.length));
            shape = tensor.shape().dims();
        }
        if (shape.length != targetShape.length) {
            throw new IllegalArgumentException("cannot align " + Arrays.toString(shape)
                    + " to " + Arrays.toString(targetShape));
        }
        boolean needsExpand = false;
        for (int axis = 0; axis < shape.length; axis++) {
            if (shape[axis] != targetShape[axis]) {
                needsExpand = true;
                break;
            }
        }
        if (needsExpand) {
            tensor = tensor.expand(targetShape);
        }
        return tensor;
    }
}
