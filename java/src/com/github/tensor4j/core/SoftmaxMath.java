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
 * Softmax reference math (tinygrad {@code Tensor._softmax} decomposition).
 */
public final class SoftmaxMath {

    private SoftmaxMath() {
    }

    public static int normalizeAxis(int axis, int rank) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("axis out of range: " + axis);
        }
        return axis;
    }

    public static Tensor softmax(Tensor input, int axis) {
        int rank = input.shape().dims().length;
        axis = normalizeAxis(axis, rank);
        float[] data = input.toFlatArray();
        int[] shape = input.shape().dims();
        int[] keepShape = shape.clone();
        keepShape[axis] = 1;
        float[] maxCompact = maxAxisInto(data, shape, axis, true);
        float[] maxBroadcast = broadcastKeepdim(maxCompact, keepShape, shape);
        float[] shifted = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            shifted[i] = data[i] - maxBroadcast[i];
        }
        float[] exp = expInto(shifted, shifted.length);
        float[] sumCompact = sumAxisInto(exp, shape, axis, true);
        float[] sumBroadcast = broadcastKeepdim(sumCompact, keepShape, shape);
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = exp[i] / sumBroadcast[i];
        }
        return Tensor.of(out, shape);
    }

    /** tinygrad {@code log_softmax}: {@code m - log(sum(exp(m)))}. */
    public static Tensor logSoftmax(Tensor input, int axis) {
        int rank = input.shape().dims().length;
        axis = normalizeAxis(axis, rank);
        float[] data = input.toFlatArray();
        int[] shape = input.shape().dims();
        int[] keepShape = shape.clone();
        keepShape[axis] = 1;
        float[] maxCompact = maxAxisInto(data, shape, axis, true);
        float[] maxBroadcast = broadcastKeepdim(maxCompact, keepShape, shape);
        float[] shifted = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            shifted[i] = data[i] - maxBroadcast[i];
        }
        float[] exp = expInto(shifted, shifted.length);
        float[] sumCompact = sumAxisInto(exp, shape, axis, true);
        float[] logSumBroadcast = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            int[] idx = Strides.unravel(i, shape);
            int[] mapped = idx.clone();
            mapped[axis] = 0;
            float sum = sumCompact[Strides.ravel(mapped, keepShape)];
            logSumBroadcast[i] = (float) Math.log(sum);
        }
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = shifted[i] - logSumBroadcast[i];
        }
        return Tensor.of(out, shape);
    }

    public static Tensor gradInput(Tensor gradOutput, Tensor softmaxOutput, int axis) {
        int rank = softmaxOutput.shape().dims().length;
        axis = normalizeAxis(axis, rank);
        float[] gradOut = gradOutput.toFlatArray();
        float[] y = softmaxOutput.toFlatArray();
        int[] shape = softmaxOutput.shape().dims();
        int[] keepShape = shape.clone();
        keepShape[axis] = 1;
        float[] dot = new float[gradOut.length];
        for (int i = 0; i < dot.length; i++) {
            dot[i] = gradOut[i] * y[i];
        }
        float[] sumDotCompact = sumAxisInto(dot, shape, axis, true);
        float[] sumDotBroadcast = broadcastKeepdim(sumDotCompact, keepShape, shape);
        float[] out = new float[gradOut.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = y[i] * (gradOut[i] - sumDotBroadcast[i]);
        }
        return Tensor.of(out, shape);
    }

    public static Tensor logSoftmaxGradInput(Tensor gradOutput, Tensor logSoftmaxOutput, int axis) {
        int rank = logSoftmaxOutput.shape().dims().length;
        axis = normalizeAxis(axis, rank);
        float[] gradOut = gradOutput.toFlatArray();
        float[] z = logSoftmaxOutput.toFlatArray();
        int[] shape = logSoftmaxOutput.shape().dims();
        int[] keepShape = shape.clone();
        keepShape[axis] = 1;
        float[] expZ = expInto(z, z.length);
        float[] sumGradCompact = sumAxisInto(gradOut, shape, axis, true);
        float[] sumGradBroadcast = broadcastKeepdim(sumGradCompact, keepShape, shape);
        float[] out = new float[gradOut.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = gradOut[i] - expZ[i] * sumGradBroadcast[i];
        }
        return Tensor.of(out, shape);
    }

    public static Tensor sumAxis(Tensor input, int axis, boolean keepdim) {
        int rank = input.shape().dims().length;
        axis = normalizeAxis(axis, rank);
        float[] data = input.toFlatArray();
        int[] shape = input.shape().dims();
        float[] reduced = sumAxisInto(data, shape, axis, keepdim);
        return Tensor.of(reduced, axisShape(shape, axis, keepdim));
    }

    public static Tensor maxAxis(Tensor input, int axis, boolean keepdim) {
        int rank = input.shape().dims().length;
        axis = normalizeAxis(axis, rank);
        float[] data = input.toFlatArray();
        int[] shape = input.shape().dims();
        float[] reduced = maxAxisInto(data, shape, axis, keepdim);
        return Tensor.of(reduced, axisShape(shape, axis, keepdim));
    }

    public static float[] maxAxisInto(float[] input, int[] shape, int axis, boolean keepdim) {
        int[] outShape = axisShape(shape, axis, keepdim);
        float[] out = new float[Strides.numel(outShape)];
        for (int i = 0; i < out.length; i++) {
            out[i] = Float.NEGATIVE_INFINITY;
        }
        int n = input.length;
        for (int flat = 0; flat < n; flat++) {
            int[] idx = Strides.unravel(flat, shape);
            int outFlat = ravelReduced(idx, shape, axis, keepdim);
            out[outFlat] = Math.max(out[outFlat], input[flat]);
        }
        return out;
    }

    public static float[] sumAxisInto(float[] input, int[] shape, int axis, boolean keepdim) {
        int[] outShape = axisShape(shape, axis, keepdim);
        float[] out = new float[Strides.numel(outShape)];
        int n = input.length;
        for (int flat = 0; flat < n; flat++) {
            int[] idx = Strides.unravel(flat, shape);
            int outFlat = ravelReduced(idx, shape, axis, keepdim);
            out[outFlat] += input[flat];
        }
        return out;
    }

    public static float[] expInto(float[] input, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.exp(input[i]);
        }
        return out;
    }

    public static float[] broadcastKeepdim(float[] reduced, int[] reducedShape, int[] targetShape) {
        float[] out = new float[Strides.numel(targetShape)];
        for (int flat = 0; flat < out.length; flat++) {
            int[] idx = Strides.unravel(flat, targetShape);
            int[] mapped = idx.clone();
            for (int axis = 0; axis < mapped.length; axis++) {
                if (reducedShape[axis] == 1 && targetShape[axis] > 1) {
                    mapped[axis] = 0;
                }
            }
            out[flat] = reduced[Strides.ravel(mapped, reducedShape)];
        }
        return out;
    }

    private static int[] axisShape(int[] shape, int axis, boolean keepdim) {
        if (keepdim) {
            int[] out = shape.clone();
            out[axis] = 1;
            return out;
        }
        return reduceShape(shape, axis);
    }

    private static int ravelReduced(int[] idx, int[] shape, int axis, boolean keepdim) {
        if (keepdim) {
            int[] mapped = idx.clone();
            mapped[axis] = 0;
            return Strides.ravel(mapped, axisShape(shape, axis, true));
        }
        return Strides.ravel(dropAxis(idx, axis), reduceShape(shape, axis));
    }

    private static int[] reduceShape(int[] shape, int axis) {
        int[] out = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                out[j++] = shape[i];
            }
        }
        return out;
    }

    private static int[] dropAxis(int[] idx, int axis) {
        int[] out = new int[idx.length - 1];
        for (int i = 0, j = 0; i < idx.length; i++) {
            if (i != axis) {
                out[j++] = idx[i];
            }
        }
        return out;
    }
}
