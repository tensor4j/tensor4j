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

/** Shape helpers for layer/group norm (PyTorch-style {@code normalized_shape}). */
public final class NormShape {

    private NormShape() {
    }

    public static void validateNormalizedShape(int[] inputShape, int[] normalizedShape) {
        if (normalizedShape.length == 0 || normalizedShape.length > inputShape.length) {
            throw new IllegalArgumentException("invalid normalized_shape for input rank");
        }
        int offset = inputShape.length - normalizedShape.length;
        for (int i = 0; i < normalizedShape.length; i++) {
            if (inputShape[offset + i] != normalizedShape[i]) {
                throw new IllegalArgumentException("normalized_shape must match trailing input dims");
            }
        }
    }

    public static int normSize(int[] normalizedShape) {
        return Strides.numel(normalizedShape);
    }

    public static int numSlices(int[] inputShape, int[] normalizedShape) {
        return Strides.numel(inputShape) / normSize(normalizedShape);
    }

    public static int[] collapsedShape(int[] inputShape, int[] normalizedShape) {
        return new int[] {numSlices(inputShape, normalizedShape), normSize(normalizedShape)};
    }

    public static int sliceOf(int flat, int[] inputShape, int[] normalizedShape) {
        return flat / normSize(normalizedShape);
    }

    public static int affineIndex(int flat, int[] normalizedShape) {
        return flat % normSize(normalizedShape);
    }

    public static int[] affineBroadcastShape(int[] inputShape, int[] normalizedShape) {
        int rank = inputShape.length;
        int[] out = new int[rank];
        Arrays.fill(out, 1);
        System.arraycopy(normalizedShape, 0, out, rank - normalizedShape.length, normalizedShape.length);
        return out;
    }

    public static int normalizeAxis(int axis, int rank) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("channel axis out of range: " + axis);
        }
        return axis;
    }

    public static int[] groupCollapsedShape(int[] inputShape, int channelAxis, int numGroups) {
        int rank = inputShape.length;
        channelAxis = normalizeAxis(channelAxis, rank);
        int channels = inputShape[channelAxis];
        if (channels % numGroups != 0) {
            throw new IllegalArgumentException("channels must divide numGroups");
        }
        int leading = 1;
        for (int i = 0; i < channelAxis; i++) {
            leading *= inputShape[i];
        }
        int trailing = 1;
        for (int i = channelAxis + 1; i < rank; i++) {
            trailing *= inputShape[i];
        }
        int normSize = (channels / numGroups) * trailing;
        return new int[] {leading, numGroups, normSize};
    }

    public static int groupKey(int flat, int[] inputShape, int channelAxis, int numGroups) {
        int[] idx = Strides.unravel(flat, inputShape);
        channelAxis = normalizeAxis(channelAxis, inputShape.length);
        int leading = 0;
        int leadingStride = 1;
        for (int i = channelAxis - 1; i >= 0; i--) {
            leading += idx[i] * leadingStride;
            leadingStride *= inputShape[i];
        }
        int group = idx[channelAxis] / (inputShape[channelAxis] / numGroups);
        return leading * numGroups + group;
    }

    public static int channelIndex(int flat, int[] inputShape, int channelAxis) {
        int[] idx = Strides.unravel(flat, inputShape);
        return idx[normalizeAxis(channelAxis, inputShape.length)];
    }

    public static int[] channelBroadcastShape(int[] inputShape, int channelAxis, int channels) {
        int rank = inputShape.length;
        channelAxis = normalizeAxis(channelAxis, rank);
        int[] out = new int[rank];
        Arrays.fill(out, 1);
        out[channelAxis] = channels;
        return out;
    }

    public static int[] nchwNormalizedShape(int[] nchw) {
        return new int[] {nchw[1], nchw[2], nchw[3]};
    }
}
