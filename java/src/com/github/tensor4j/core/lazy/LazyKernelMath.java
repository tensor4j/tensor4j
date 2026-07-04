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
import java.util.List;
import java.util.Map;

/** Evaluates fused elementwise {@link LazyKernel} bodies in one flat pass. */
final class LazyKernelMath {

    private LazyKernelMath() {
    }

    static Tensor evalFusedBody(List<LazyUOp> body, Map<LazyUOp, Tensor> memo) {
        Map<LazyUOp, float[]> flats = new HashMap<>();
        for (LazyUOp node : body) {
            flats.put(node, evalFlat(node, flats, memo));
        }
        LazyUOp output = body.get(body.size() - 1);
        return Tensor.of(flats.get(output), LazyUOpShapes.infer(output));
    }

    private static float[] evalFlat(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo) {
        int[] shape = LazyUOpShapes.infer(node);
        int n = Strides.numel(shape);
        return switch (node.op()) {
            case BUFFER -> readBuffer(node, memo);
            case ADD -> evalAdd(node, flats, memo, n);
            case MUL -> evalMul(node, flats, memo, n);
            case SUB -> evalSub(node, flats, memo, n);
            case DIV -> evalDiv(node, flats, memo, n);
            case NEG -> evalNeg(node, flats, memo, n);
            case RECIP -> evalRecip(node, flats, memo, n);
            case RELU -> evalRelu(node, flats, memo, n);
            case POW -> evalPow(node, flats, memo, n);
            case LOG2 -> evalLog2(node, flats, memo, n);
            case EXP2 -> evalExp2(node, flats, memo, n);
            case EXP -> evalExp(node, flats, memo, n);
            case SQRT -> evalSqrt(node, flats, memo, n);
            case MAX -> evalMax(node, flats, memo, n);
            case GT_MASK -> evalGtMask(node, flats, memo, n);
            case EQ_MASK -> evalEqMask(node, flats, memo, n);
            default -> throw new IllegalStateException("unsupported fused op " + node.op());
        };
    }

    private static float[] readBuffer(LazyUOp node, Map<LazyUOp, Tensor> memo) {
        Tensor cached = memo.get(node);
        if (cached != null) {
            return cached.toFlatArray();
        }
        return node.buffer().toFlatArray();
    }

    private static float[] readOperand(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int[] targetShape) {
        Tensor tensor;
        if (flats.containsKey(node)) {
            tensor = Tensor.of(flats.get(node), LazyUOpShapes.infer(node));
        } else if (memo.containsKey(node)) {
            tensor = memo.get(node);
        } else if (node.op() == LazyUOp.Kind.BUFFER) {
            tensor = node.buffer();
        } else {
            tensor = LazyRealizer.realize(node, false, memo);
        }
        return LazyRealizer.expandToShape(tensor, targetShape).toFlatArray();
    }

    private static float[] evalAdd(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] left = readOperand(node.src(0), flats, memo, shape);
        float[] right = readOperand(node.src(1), flats, memo, shape);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = left[i] + right[i];
        }
        return out;
    }

    private static float[] evalMul(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] left = readOperand(node.src(0), flats, memo, shape);
        float[] right = readOperand(node.src(1), flats, memo, shape);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = left[i] * right[i];
        }
        return out;
    }

    private static float[] evalSub(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] left = readOperand(node.src(0), flats, memo, shape);
        float[] right = readOperand(node.src(1), flats, memo, shape);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = left[i] - right[i];
        }
        return out;
    }

    private static float[] evalDiv(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] left = readOperand(node.src(0), flats, memo, shape);
        float[] right = readOperand(node.src(1), flats, memo, shape);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = left[i] / right[i];
        }
        return out;
    }

    private static float[] evalRecip(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] input = readOperand(node.src(0), flats, memo, shape);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = 1f / input[i];
        }
        return out;
    }

    private static float[] evalNeg(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] input = readOperand(node.src(0), flats, memo, shape);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = -input[i];
        }
        return out;
    }

    private static float[] evalRelu(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] input = readOperand(node.src(0), flats, memo, shape);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = Math.max(0f, input[i]);
        }
        return out;
    }

    private static float[] evalPow(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] input = readOperand(node.src(0), flats, memo, shape);
        return LazyMath.powInto(input, LazyMath.floatArg(node.arg()[0]), n);
    }

    private static float[] evalLog2(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] input = readOperand(node.src(0), flats, memo, shape);
        return LazyMath.log2Into(input, n);
    }

    private static float[] evalExp2(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] input = readOperand(node.src(0), flats, memo, shape);
        return LazyMath.exp2Into(input, n);
    }

    private static float[] evalExp(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] input = readOperand(node.src(0), flats, memo, shape);
        return LazyMath.expInto(input, n);
    }

    private static float[] evalSqrt(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] input = readOperand(node.src(0), flats, memo, shape);
        return LazyMath.sqrtInto(input, n);
    }

    private static float[] evalMax(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] left = readOperand(node.src(0), flats, memo, shape);
        float[] right = readOperand(node.src(1), flats, memo, shape);
        return LazyMath.maxInto(left, right, n);
    }

    private static float[] evalGtMask(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] left = readOperand(node.src(0), flats, memo, shape);
        float[] right = readOperand(node.src(1), flats, memo, shape);
        return LazyMath.gtMaskInto(left, right, n);
    }

    private static float[] evalEqMask(LazyUOp node, Map<LazyUOp, float[]> flats, Map<LazyUOp, Tensor> memo, int n) {
        int[] shape = LazyUOpShapes.infer(node);
        float[] left = readOperand(node.src(0), flats, memo, shape);
        float[] right = readOperand(node.src(1), flats, memo, shape);
        return LazyMath.eqMaskInto(left, right, n);
    }
}
