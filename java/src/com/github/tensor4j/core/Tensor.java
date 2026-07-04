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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import com.github.tensor4j.autograd.AddFunction;
import com.github.tensor4j.autograd.DivFunction;
import com.github.tensor4j.autograd.ExpandFunction;
import com.github.tensor4j.autograd.Function;
import com.github.tensor4j.autograd.MatMulFunction;
import com.github.tensor4j.autograd.MeanFunction;
import com.github.tensor4j.autograd.MulFunction;
import com.github.tensor4j.autograd.NegFunction;
import com.github.tensor4j.autograd.PermuteFunction;
import com.github.tensor4j.autograd.ReluFunction;
import com.github.tensor4j.autograd.ReshapeFunction;
import com.github.tensor4j.autograd.SumFunction;
import com.github.tensor4j.autograd.LogSoftmaxFunction;
import com.github.tensor4j.autograd.Conv2dFunction;
import com.github.tensor4j.autograd.ConvTranspose2dFunction;
import com.github.tensor4j.autograd.Pool2dFunction;
import com.github.tensor4j.autograd.AdaptivePool2dFunction;
import com.github.tensor4j.autograd.BatchNorm2dEvalFunction;
import com.github.tensor4j.autograd.BatchNorm2dTrainFunction;
import com.github.tensor4j.autograd.DropoutFunction;
import com.github.tensor4j.autograd.GroupNormFunction;
import com.github.tensor4j.autograd.LayerNormFunction;
import com.github.tensor4j.autograd.MaxUnpool2dFunction;
import com.github.tensor4j.autograd.SoftmaxFunction;

/**
 * Float32 tensor: {@link StorageBuffer} + {@link TensorLayout} (tinygrad buffer + movement metadata).
 */
public final class Tensor {

    private final StorageBuffer buffer;
    private final TensorLayout layout;
    private boolean requiresGrad;
    private Tensor grad;
    private Function creator;
    private final List<Tensor> children = new ArrayList<>();

    private Tensor(StorageBuffer buffer, TensorLayout layout, boolean requiresGrad) {
        this.buffer = buffer;
        this.layout = layout;
        this.requiresGrad = requiresGrad;
    }

    public static Tensor of(float value) {
        return new Tensor(StorageBuffer.wrapOwned(new float[] {value}), TensorLayout.contiguous(new int[] {1}), false);
    }

    public static Tensor of(float[] values, int... dims) {
        int[] shape = dims.clone();
        if (Strides.numel(shape) != values.length) {
            throw new IllegalArgumentException("data length " + values.length + " != shape " + Strides.numel(shape));
        }
        return new Tensor(StorageBuffer.wrapOwned(Arrays.copyOf(values, values.length)), TensorLayout.contiguous(shape), false);
    }

    public static Tensor zeros(int... dims) {
        int[] shape = dims.clone();
        return new Tensor(StorageBuffer.allocate(Strides.numel(shape)), TensorLayout.contiguous(shape), false);
    }

    public static Tensor randn(int... dims) {
        int[] shape = dims.clone();
        float[] data = new float[Strides.numel(shape)];
        for (int i = 0; i < data.length; i++) {
            double u1 = Math.max(1e-9, Math.random());
            double u2 = Math.random();
            data[i] = (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
        }
        return new Tensor(StorageBuffer.wrapOwned(data), TensorLayout.contiguous(shape), false);
    }

    Tensor view(TensorLayout nextLayout) {
        return new Tensor(buffer, nextLayout, requiresGrad);
    }

    public Tensor withGrad(boolean enabled) {
        requiresGrad = enabled;
        return this;
    }

    public Shape shape() {
        return new Shape(layout.shape());
    }

    public StorageBuffer buffer() {
        return buffer;
    }

    public TensorLayout layout() {
        return layout;
    }

    public int numel() {
        return layout.numel();
    }

    public boolean isContiguous() {
        return layout.isContiguous();
    }

    public int[] strides() {
        return layout.strides();
    }

    public float[] data() {
        if (hasDirectContiguousStorage()) {
            return buffer.data();
        }
        return toFlatArray();
    }

    public float[] mutableData() {
        if (!hasDirectContiguousStorage()) {
            throw new IllegalStateException("tensor is not a writable contiguous buffer");
        }
        return buffer.data();
    }

    public float get(int... indices) {
        return buffer.data()[Strides.offset(buffer.offset(), indices, layout.strides())];
    }

    public void set(float value, int... indices) {
        buffer.data()[Strides.offset(buffer.offset(), indices, layout.strides())] = value;
    }

    public float getFlat(int flatIndex) {
        return Strides.read(buffer.data(), buffer.offset(), layout.shape(), layout.strides(), flatIndex);
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public Tensor grad() {
        return grad;
    }

    public void setGrad(Tensor value) {
        grad = value;
    }

    public void setCreator(Function function, Tensor... parents) {
        creator = function;
        children.clear();
        children.addAll(Arrays.asList(parents));
    }

    public Function graphCreator() {
        return creator;
    }

    public Tensor sub(Tensor other) {
        return add(other.neg());
    }

    public Tensor neg() {
        int n = numel();
        float[] result = new float[n];
        if (hasDirectContiguousStorage()) {
            TensorMath.negInto(result, buffer.data(), n);
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = -getFlat(i);
            }
        }
        return attachUnary(result, new NegFunction(this));
    }

    public void applyGrad(float learningRate) {
        if (grad == null) {
            return;
        }
        if (!hasDirectContiguousStorage()) {
            throw new IllegalStateException("applyGrad requires contiguous owned parameters");
        }
        float[] gradient = grad.contiguous().toFlatArray();
        TensorMath.axpy(buffer.data(), gradient, numel(), learningRate);
        grad = null;
    }

    public Tensor add(Tensor other) {
        requireSameShape(other);
        int n = numel();
        float[] result = new float[n];
        if (hasDirectContiguousStorage() && other.hasDirectContiguousStorage()) {
            TensorMath.addInto(result, buffer.data(), other.buffer.data(), n);
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = getFlat(i) + other.getFlat(i);
            }
        }
        return attachBinary(result, other, new AddFunction(this, other));
    }

    public Tensor mul(Tensor other) {
        requireSameShape(other);
        int n = numel();
        float[] result = new float[n];
        if (hasDirectContiguousStorage() && other.hasDirectContiguousStorage()) {
            TensorMath.mulInto(result, buffer.data(), other.buffer.data(), n);
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = getFlat(i) * other.getFlat(i);
            }
        }
        return attachBinary(result, other, new MulFunction(this, other));
    }

    public Tensor div(Tensor other) {
        requireSameShape(other);
        int n = numel();
        float[] result = new float[n];
        if (hasDirectContiguousStorage() && other.hasDirectContiguousStorage()) {
            TensorMath.divInto(result, buffer.data(), other.buffer.data(), n);
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = getFlat(i) / other.getFlat(i);
            }
        }
        return attachBinary(result, other, new DivFunction(this, other));
    }

    public Tensor matmul(Tensor other) {
        if (layout.rank() != 2 || other.layout.rank() != 2) {
            throw new IllegalArgumentException("matmul requires rank-2 tensors");
        }
        int rows = layout.dim(0);
        int inner = layout.dim(1);
        int cols = other.layout.dim(1);
        if (inner != other.layout.dim(0)) {
            throw new IllegalArgumentException("inner dimensions must match for matmul");
        }
        float[] left = contiguousFlatOrCopy();
        float[] right = other.contiguousFlatOrCopy();
        float[] result = TensorMath.matmul2d(left, rows, inner, right, cols);
        Tensor out = Tensor.of(result, rows, cols);
        if (requiresGrad || other.requiresGrad) {
            out.withGrad(true);
            out.setCreator(new MatMulFunction(this, other), this, other);
        }
        return out;
    }

    public Tensor transpose2d() {
        if (layout.rank() != 2) {
            throw new IllegalArgumentException("transpose2d requires rank-2 tensor");
        }
        return permute(1, 0);
    }

    public Tensor permute(int... order) {
        if (order.length != layout.rank()) {
            throw new IllegalArgumentException("permute order must match tensor rank");
        }
        int[] normalized = normalizeAxisOrder(order);
        Tensor out = view(layout.permute(normalized));
        if (requiresGrad) {
            out.withGrad(true);
            out.setCreator(new PermuteFunction(this, normalized), this);
        }
        return out;
    }

    /** Broadcast view via stride-0 axes (tinygrad {@code expand}). */
    public Tensor expand(int... targetShape) {
        int[] shape = targetShape.clone();
        int[] inputShape = layout.shape();
        Tensor out = view(layout.expand(shape));
        if (requiresGrad) {
            out.withGrad(true);
            out.setCreator(new ExpandFunction(this, inputShape), this);
        }
        return out;
    }

    public Tensor flatten() {
        return reshape(numel());
    }

    public Tensor sum() {
        float total = Strides.sum(buffer.data(), buffer.offset(), layout.shape(), layout.strides());
        Tensor out = Tensor.of(total).withGrad(requiresGrad);
        if (requiresGrad) {
            out.setCreator(new SumFunction(this), this);
        }
        return out;
    }

    public Tensor sumAxis(int axis) {
        int rank = layout.rank();
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("axis out of range: " + axis);
        }
        int[] inShape = layout.shape();
        int[] outShape = new int[rank - 1];
        for (int i = 0, j = 0; i < rank; i++) {
            if (i != axis) {
                outShape[j++] = inShape[i];
            }
        }
        float[] acc = new float[Strides.numel(outShape)];
        int n = numel();
        for (int flat = 0; flat < n; flat++) {
            int[] indices = Strides.unravel(flat, inShape);
            int[] reduced = new int[outShape.length];
            for (int i = 0, j = 0; i < rank; i++) {
                if (i != axis) {
                    reduced[j++] = indices[i];
                }
            }
            acc[Strides.ravel(reduced, outShape)] += getFlat(flat);
        }
        return Tensor.of(acc, outShape);
    }

    public Tensor relu() {
        int n = numel();
        float[] result = new float[n];
        if (hasDirectContiguousStorage()) {
            TensorMath.reluInto(result, buffer.data(), n);
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = Math.max(0f, getFlat(i));
            }
        }
        return attachUnary(result, new ReluFunction(this));
    }

    public Tensor mean() {
        float mean = TensorMath.mean(toFlatArray(), numel());
        Tensor out = Tensor.of(mean).withGrad(requiresGrad);
        if (requiresGrad) {
            out.setCreator(new MeanFunction(this), this);
        }
        return out;
    }

    public Tensor pow2() {
        return mul(this);
    }

    /** Softmax along axis (tinygrad {@code Tensor.softmax}). Default axis is last. */
    public Tensor softmax() {
        return softmax(-1);
    }

    public Tensor softmax(int axis) {
        if (!requiresGrad) {
            return SoftmaxMath.softmax(this, axis);
        }
        SoftmaxFunction function = new SoftmaxFunction(this, axis);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this);
        return out;
    }

    public Tensor logSoftmax() {
        return logSoftmax(-1);
    }

    public Tensor logSoftmax(int axis) {
        if (!requiresGrad) {
            return SoftmaxMath.logSoftmax(this, axis);
        }
        LogSoftmaxFunction function = new LogSoftmaxFunction(this, axis);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this);
        return out;
    }

    /** NCHW conv2d (tinygrad {@code Tensor.conv2d} teaching subset — no Winograd). */
    public Tensor conv2d(Tensor weight) {
        return conv2d(weight, Conv2dMath.defaultArg());
    }

    public Tensor conv2d(Tensor weight, int[] arg) {
        Tensor out = Conv2dMath.forward(this, weight, arg);
        if (requiresGrad || weight.requiresGrad()) {
            out.withGrad(true);
            out.setCreator(new Conv2dFunction(this, weight, arg), this, weight);
        }
        return out;
    }

    /** Depthwise NCHW conv (tinygrad {@code conv2d(..., groups=C)}). Weight {@code [C, 1, kH, kW]}. */
    public Tensor depthwiseConv2d(Tensor weight) {
        return depthwiseConv2d(weight, Conv2dMath.defaultArg());
    }

    public Tensor depthwiseConv2d(Tensor weight, int[] arg) {
        return conv2d(weight, Conv2dMath.depthwiseArg(arg, shape().dims()[1]));
    }

    public Tensor convTranspose2d(Tensor weight) {
        return convTranspose2d(weight, ConvTranspose2dMath.defaultArg());
    }

    public Tensor convTranspose2d(Tensor weight, int[] arg) {
        Tensor out = ConvTranspose2dMath.forward(this, weight, arg);
        if (requiresGrad || weight.requiresGrad()) {
            out.withGrad(true);
            out.setCreator(new ConvTranspose2dFunction(this, weight, arg), this, weight);
        }
        return out;
    }

    public Tensor maxPool2d(int kernel, int stride) {
        return pool2d(Pool2dArg.maxPacked(kernel, stride));
    }

    public Tensor avgPool2d(int kernel, int stride) {
        return pool2d(Pool2dArg.avgPacked(kernel, stride));
    }

    public Tensor pool2d(int[] arg) {
        if (!requiresGrad) {
            return Pool2dMath.forward(this, arg);
        }
        Pool2dFunction function = new Pool2dFunction(this, arg);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this);
        return out;
    }

    public MaxPool2dResult maxPool2dWithIndices(int kernel, int stride) {
        int[] arg = Pool2dArg.maxPacked(kernel, stride);
        Pool2dMath.ForwardResult meta = Pool2dMath.forwardWithMeta(this, arg);
        float[] idx = new float[meta.flatSpatialIndex.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = meta.flatSpatialIndex[i];
        }
        Tensor indices = Tensor.of(idx, meta.output.shape().dims());
        Tensor output = meta.output;
        if (requiresGrad) {
            Pool2dFunction function = new Pool2dFunction(this, arg);
            output = function.forward();
            output.withGrad(true);
            output.setCreator(function, this);
        }
        return new MaxPool2dResult(output, indices);
    }

    public Tensor maxUnpool2d(Tensor indices, int[] poolArg, int[] outputShape) {
        Tensor out = MaxUnpool2dMath.forward(this, indices, poolArg, outputShape);
        if (requiresGrad) {
            out.withGrad(true);
            out.setCreator(new MaxUnpool2dFunction(this, indices, poolArg, outputShape), this);
        }
        return out;
    }

    public Tensor adaptiveAvgPool2d(int outH, int outW) {
        return adaptivePool2d(AdaptivePool2dArg.packed(outH, outW, Pool2dArg.MODE_AVG));
    }

    public Tensor adaptiveMaxPool2d(int outH, int outW) {
        return adaptivePool2d(AdaptivePool2dArg.packed(outH, outW, Pool2dArg.MODE_MAX));
    }

    public Tensor adaptivePool2d(int[] arg) {
        if (!requiresGrad) {
            return AdaptivePool2dMath.forward(this, arg);
        }
        AdaptivePool2dFunction function = new AdaptivePool2dFunction(this, arg);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this);
        return out;
    }

    /** Inference batch norm with per-channel {@code weight}, {@code bias}, {@code mean}, {@code var}. */
    public Tensor batchNorm2d(Tensor weight, Tensor bias, Tensor mean, Tensor var, float eps) {
        Tensor out = BatchNorm2dMath.forwardEval(this, weight, bias, mean, var, eps);
        if (requiresGrad || weight.requiresGrad() || bias.requiresGrad()) {
            out.withGrad(true);
            out.setCreator(new BatchNorm2dEvalFunction(this, weight, bias, mean, var, eps),
                    this, weight, bias);
        }
        return out;
    }

    /** Training batch norm (batch statistics). */
    public Tensor batchNorm2dTrain(Tensor weight, Tensor bias, float eps) {
        if (!requiresGrad && !weight.requiresGrad() && !bias.requiresGrad()) {
            return BatchNorm2dMath.forwardTrain(this, weight, bias, eps).output;
        }
        BatchNorm2dTrainFunction function = new BatchNorm2dTrainFunction(this, weight, bias, eps);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this, weight, bias);
        return out;
    }

    /** Layer norm over trailing {@code normalizedShape}; {@code weight}/{@code bias} match that shape. */
    public Tensor layerNorm(Tensor weight, Tensor bias, int[] normalizedShape, float eps) {
        if (!requiresGrad && !weight.requiresGrad() && !bias.requiresGrad()) {
            return LayerNormMath.forward(this, weight, bias, normalizedShape, eps).output;
        }
        LayerNormFunction function = new LayerNormFunction(this, weight, bias, normalizedShape, eps);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this, weight, bias);
        return out;
    }

    /** NCHW convenience: {@code normalizedShape = [C,H,W]}. */
    public Tensor layerNorm2d(Tensor weight, Tensor bias, float eps) {
        return layerNorm(weight, bias, NormShape.nchwNormalizedShape(shape().dims()), eps);
    }

    /** Group norm over {@code (C/G, *trailing)} at {@code channelAxis}; {@code weight}/{@code bias} length {@code C}. */
    public Tensor groupNorm(int numGroups, Tensor weight, Tensor bias, int channelAxis, float eps) {
        if (!requiresGrad && !weight.requiresGrad() && !bias.requiresGrad()) {
            return GroupNormMath.forward(this, weight, bias, numGroups, channelAxis, eps).output;
        }
        GroupNormFunction function = new GroupNormFunction(this, weight, bias, numGroups, channelAxis, eps);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this, weight, bias);
        return out;
    }

    /** NCHW convenience: {@code channelAxis = 1}. */
    public Tensor groupNorm2d(int numGroups, Tensor weight, Tensor bias, float eps) {
        return groupNorm(numGroups, weight, bias, 1, eps);
    }

    /** Training dropout; samples a Bernoulli mask automatically. */
    public Tensor dropout(float p) {
        return dropout(p, null);
    }

    /** Training dropout with optional {@code seed} for reproducibility. */
    public Tensor dropout(float p, Long seed) {
        if (!requiresGrad) {
            Tensor mask = seed == null
                    ? DropoutMath.sampleMask(shape().dims(), p)
                    : DropoutMath.sampleMask(shape().dims(), p, seed);
            return DropoutMath.forward(this, mask, p);
        }
        DropoutFunction function = new DropoutFunction(this, p, seed);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this);
        return out;
    }

    /** Dropout with an explicit 0/1 {@code mask} (advanced / parity). */
    public Tensor dropoutWithMask(Tensor mask, float p) {
        if (!requiresGrad) {
            return DropoutMath.forward(this, mask, p);
        }
        DropoutFunction function = new DropoutFunction(this, mask, p);
        Tensor out = function.forward();
        out.withGrad(true);
        out.setCreator(function, this);
        return out;
    }

    public void zeroGrad() {
        grad = null;
    }

    public void backward() {
        com.github.tensor4j.autograd.AutogradEngine.backward(this);
    }

    public void backward(Tensor gradOutput) {
        com.github.tensor4j.autograd.AutogradEngine.backward(this, gradOutput);
    }

    public Tensor detach() {
        return Tensor.of(toFlatArray(), layout.shape()).withGrad(false);
    }

    public Tensor reshape(int... dims) {
        int[] newShape = dims.clone();
        if (Strides.numel(newShape) != numel()) {
            throw new IllegalArgumentException("reshape incompatible element counts");
        }
        if (!isContiguous()) {
            return contiguous().reshape(dims);
        }
        int[] inputShape = layout.shape();
        Tensor out = view(layout.reshape(newShape));
        if (requiresGrad) {
            out.withGrad(true);
            out.setCreator(new ReshapeFunction(this, inputShape), this);
        }
        return out;
    }

    public Tensor contiguous() {
        if (hasDirectContiguousStorage()) {
            return this;
        }
        Tensor dense = Tensor.of(toFlatArray(), layout.shape()).withGrad(requiresGrad);
        if (requiresGrad) {
            dense.setCreator(new com.github.tensor4j.autograd.ContiguousFunction(this), this);
        }
        return dense;
    }

    public float[] toFlatArray() {
        int n = numel();
        float[] flat = new float[n];
        for (int i = 0; i < n; i++) {
            flat[i] = getFlat(i);
        }
        return flat;
    }

    public float[] contiguousFlatOrCopy() {
        if (hasDirectContiguousStorage()) {
            float[] flat = new float[numel()];
            System.arraycopy(buffer.data(), buffer.offset(), flat, 0, numel());
            return flat;
        }
        return toFlatArray();
    }

    boolean hasDirectContiguousStorage() {
        return isContiguous()
                && buffer.isOwnedRoot()
                && buffer.offset() == 0
                && buffer.size() == numel();
    }

    private Tensor attachUnary(float[] result, Function autogradNode) {
        Tensor out = Tensor.of(result, layout.shape());
        if (requiresGrad) {
            out.withGrad(true);
            out.setCreator(autogradNode, this);
        }
        return out;
    }

    private Tensor attachBinary(float[] result, Tensor other, Function autogradNode) {
        Tensor out = Tensor.of(result, layout.shape());
        if (requiresGrad || other.requiresGrad) {
            out.withGrad(true);
            out.setCreator(autogradNode, this, other);
        }
        return out;
    }

    private void requireSameShape(Tensor other) {
        if (!Arrays.equals(layout.shape(), other.layout.shape())) {
            throw new IllegalArgumentException(
                    "shape mismatch: " + Arrays.toString(layout.shape()) + " vs " + Arrays.toString(other.layout.shape()));
        }
    }

    private static int[] normalizeAxisOrder(int[] order) {
        int[] normalized = order.clone();
        boolean[] used = new boolean[normalized.length];
        for (int axis : normalized) {
            if (axis < 0 || axis >= normalized.length || used[axis]) {
                throw new IllegalArgumentException("invalid permute order: " + Arrays.toString(order));
            }
            used[axis] = true;
        }
        return normalized;
    }

    @Override
    public String toString() {
        float[] preview = data();
        return "Tensor" + shape() + Arrays.toString(Arrays.copyOf(preview, Math.min(preview.length, 8)))
                + (preview.length > 8 ? "..." : "");
    }
}
