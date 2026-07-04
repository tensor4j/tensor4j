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

import com.github.tensor4j.core.AdaptivePool2dArg;
import com.github.tensor4j.core.BatchNorm2dMath;
import com.github.tensor4j.core.Conv2dMath;
import com.github.tensor4j.core.ConvTranspose2dMath;
import com.github.tensor4j.core.DropoutMath;
import com.github.tensor4j.core.NormShape;
import com.github.tensor4j.core.Pool2dArg;
import com.github.tensor4j.core.Shape;
import com.github.tensor4j.core.Strides;
import com.github.tensor4j.core.Tensor;
import java.util.Arrays;
import java.util.List;

/**
 * Lazy float32 tensor wrapping a {@link LazyUOp} DAG (tinygrad {@code Tensor.uop}).
 * Movement and elementwise ops add nodes; {@link #realize()} materializes via dependency order.
 */
public final class LazyTensor {

    private final LazyUOp uop;
    private Tensor realized;
    private boolean realizedWithGrad;
    private int[] cachedShape;

    private LazyTensor(LazyUOp uop) {
        this.uop = uop;
        LazyUOpShapes.validateOrThrow(uop);
    }

    public static LazyTensor of(float[] values, int... dims) {
        return wrap(Tensor.of(values, dims));
    }

    public static LazyTensor wrap(Tensor tensor) {
        return new LazyTensor(LazyUOp.buffer(tensor));
    }

    /** Lazy GGUF slice load (Q4_0 dequant or F32 mmap). */
    public static LazyTensor ggufLoad(LazyGgufSlice slice) {
        return new LazyTensor(LazyGgufLoad.fromSlice(slice));
    }

    /** Lazy Q4_0 dequant from mmap quant bytes (tinygrad {@code ggml_data_to_tensor}). */
    public static LazyTensor dequantQ4(LazyGgufSlice slice) {
        return ggufLoad(slice);
    }

    /** Root {@link LazyUOp} node for this tensor (tinygrad {@code Tensor.uop}). */
    public LazyUOp uop() {
        return uop;
    }

    /** Enable gradients on a leaf parameter (tinygrad {@code requires_grad=True}). */
    public LazyTensor withGrad(boolean enabled) {
        if (uop.op() != LazyUOp.Kind.BUFFER) {
            throw new IllegalArgumentException("withGrad applies to leaf LazyTensor only");
        }
        uop.buffer().withGrad(enabled);
        realized = null;
        realizedWithGrad = false;
        return this;
    }

    public boolean requiresGrad() {
        return LazyGraph.needsGrad(uop);
    }

    public int[] shape() {
        if (cachedShape == null) {
            cachedShape = LazyUOpShapes.infer(uop);
        }
        return cachedShape.clone();
    }

    public Shape typedShape() {
        return new Shape(shape());
    }

    public int numel() {
        int total = 1;
        for (int dim : shape()) {
            total *= dim;
        }
        return total;
    }

    public int graphDepth() {
        return LazyGraph.graphDepth(uop);
    }

    public int graphNodeCount() {
        return LazyGraph.nodeCount(uop);
    }

    public boolean isMovementOnly() {
        return uop.isMovementOnly();
    }

    public LazyShape movementShape() {
        return uop.movementShape();
    }

    public List<LazyUOp> toposort() {
        return LazyGraph.toposort(uop);
    }

    /** Build fused kernel schedule (tinygrad {@code create_linear_with_vars}). */
    public LazySchedule schedule() {
        return LazySchedule.build(uop);
    }

    public boolean isRealized() {
        return realized != null;
    }

    /** Schedule and materialize (tinygrad {@code realize()}). Idempotent. */
    public Tensor realize() {
        if (realized == null) {
            if (uop.op() == LazyUOp.Kind.BUFFER) {
                realized = uop.buffer();
            } else {
                realized = LazyRealizer.realize(uop);
            }
            realizedWithGrad = false;
        }
        return realized;
    }

    /**
     * Propagate loss gradient through the lazy UOp DAG (tinygrad {@code compute_gradient}).
     */
    public void backward() {
        if (numel() != 1) {
            throw new IllegalArgumentException("backward without seed requires scalar output, got numel=" + numel());
        }
        LazyAutograd.backward(uop, LazyUOp.buffer(Tensor.of(1f)));
        realize();
    }

    public void backward(LazyTensor gradOutput) {
        LazyAutograd.backward(uop, gradOutput.uop());
        realize();
    }

    /** Gradient on the leaf buffer after {@link #backward()}. */
    public Tensor grad() {
        if (uop.op() != LazyUOp.Kind.BUFFER) {
            throw new IllegalStateException("grad() applies to leaf LazyTensor; use leafTensor().grad() on outputs");
        }
        return uop.buffer().grad();
    }

    public void zeroGrad() {
        LazyGraph.zeroGrad(uop);
    }

    public LazyTensor reshape(int... dims) {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.RESHAPE, uop, dims.clone()));
    }

    public LazyTensor permute(int... order) {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.PERMUTE, uop, order.clone()));
    }

    public LazyTensor expand(int... targetShape) {
        int[] current = LazyUOpShapes.infer(uop);
        LazyUOp root = uop;
        if (current.length < targetShape.length) {
            root = LazyUOp.unary(LazyUOp.Kind.RESHAPE, uop, Strides.alignLeft(current, targetShape.length));
        }
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.EXPAND, root, targetShape.clone()));
    }

    public LazyTensor add(LazyTensor other) {
        return new LazyTensor(LazyUOp.binary(LazyUOp.Kind.ADD, uop, other.uop));
    }

    public LazyTensor mul(LazyTensor other) {
        return new LazyTensor(LazyUOp.binary(LazyUOp.Kind.MUL, uop, other.uop));
    }

    public LazyTensor sub(LazyTensor other) {
        return new LazyTensor(LazyUOp.binary(LazyUOp.Kind.SUB, uop, other.uop()));
    }

    public LazyTensor div(LazyTensor other) {
        return new LazyTensor(LazyUOp.binary(LazyUOp.Kind.DIV, uop, other.uop()));
    }

    public LazyTensor recip() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.RECIP, uop, null));
    }

    public LazyTensor neg() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.NEG, uop, null));
    }

    public LazyTensor relu() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.RELU, uop, null));
    }

    public LazyTensor sum() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.SUM, uop, null));
    }

    public LazyTensor mean() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.MEAN, uop, null));
    }

    public LazyTensor pow2() {
        return pow(2f);
    }

    public LazyTensor pow(float exponent) {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.POW, uop, new int[] {LazyMath.floatArg(exponent)}));
    }

    public LazyTensor max(LazyTensor other) {
        return new LazyTensor(LazyUOp.binary(LazyUOp.Kind.MAX, uop, other.uop()));
    }

    public LazyTensor where(LazyTensor ifTrue, LazyTensor ifFalse) {
        return new LazyTensor(LazyUOp.ternary(LazyUOp.Kind.WHERE, uop, ifTrue.uop(), ifFalse.uop()));
    }

    public LazyTensor log2() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.LOG2, uop, null));
    }

    public LazyTensor exp2() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.EXP2, uop, null));
    }

    public LazyTensor sqrt() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.SQRT, uop, null));
    }

    public LazyTensor pad(int... padArg) {
        int[] shape = LazyUOpShapes.infer(uop);
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.PAD, uop, LazyPadShrink.validatePadArg(shape, padArg)));
    }

    public LazyTensor shrink(int... shrinkArg) {
        int[] shape = LazyUOpShapes.infer(uop);
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.SHRINK, uop, LazyPadShrink.validateShrinkArg(shape, shrinkArg)));
    }

    public LazyTensor contiguous() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.CONTIGUOUS, uop, null));
    }

    public LazyTensor cast() {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.CAST, uop, null));
    }

    public LazyTensor softmax() {
        return softmax(-1);
    }

    public LazyTensor softmax(int axis) {
        return new LazyTensor(LazySoftmax.softmax(uop, axis));
    }

    public LazyTensor logSoftmax() {
        return logSoftmax(-1);
    }

    public LazyTensor logSoftmax(int axis) {
        return new LazyTensor(LazySoftmax.logSoftmax(uop, axis));
    }

    public LazyTensor conv2d(LazyTensor weight) {
        return conv2d(weight, Conv2dMath.defaultArg());
    }

    public LazyTensor conv2d(LazyTensor weight, int[] arg) {
        return new LazyTensor(LazyConv2d.conv2d(uop, weight.uop(), arg));
    }

    public LazyTensor depthwiseConv2d(LazyTensor weight) {
        return depthwiseConv2d(weight, Conv2dMath.defaultArg());
    }

    public LazyTensor depthwiseConv2d(LazyTensor weight, int[] arg) {
        return conv2d(weight, Conv2dMath.depthwiseArg(arg, shape()[1]));
    }

    public LazyTensor convTranspose2d(LazyTensor weight) {
        return convTranspose2d(weight, ConvTranspose2dMath.defaultArg());
    }

    public LazyTensor convTranspose2d(LazyTensor weight, int[] arg) {
        return new LazyTensor(LazyConvTranspose2d.convTranspose2d(uop, weight.uop(), arg));
    }

    public LazyTensor maxPool2d(int kernel, int stride) {
        return pool2d(Pool2dArg.maxPacked(kernel, stride));
    }

    public LazyTensor avgPool2d(int kernel, int stride) {
        return pool2d(Pool2dArg.avgPacked(kernel, stride));
    }

    public LazyTensor pool2d(int[] arg) {
        return new LazyTensor(LazyPool2d.pool2d(uop, arg));
    }

    public LazyTensor adaptiveAvgPool2d(int outH, int outW) {
        return adaptivePool2d(AdaptivePool2dArg.packed(outH, outW, Pool2dArg.MODE_AVG));
    }

    public LazyTensor adaptiveMaxPool2d(int outH, int outW) {
        return adaptivePool2d(AdaptivePool2dArg.packed(outH, outW, Pool2dArg.MODE_MAX));
    }

    public LazyTensor adaptivePool2d(int[] arg) {
        return new LazyTensor(LazyAdaptivePool2d.adaptivePool2d(uop, arg));
    }

    public LazyTensor batchNorm2d(LazyTensor weight, LazyTensor bias, LazyTensor mean, LazyTensor var, float eps) {
        return new LazyTensor(LazyBatchNorm2d.batchNorm2dEval(uop, weight.uop(), bias.uop(), mean.uop(), var.uop(), eps));
    }

    public LazyTensor batchNorm2dTrain(LazyTensor weight, LazyTensor bias, float eps) {
        return new LazyTensor(LazyBatchNorm2d.batchNorm2dTrain(uop, weight.uop(), bias.uop(), eps));
    }

    public LazyTensor layerNorm(LazyTensor weight, LazyTensor bias, int[] normalizedShape, float eps) {
        return new LazyTensor(LazyLayerNorm.layerNorm(uop, weight.uop(), bias.uop(), normalizedShape, eps));
    }

    public LazyTensor layerNorm2d(LazyTensor weight, LazyTensor bias, float eps) {
        return layerNorm(weight, bias, NormShape.nchwNormalizedShape(shape()), eps);
    }

    public LazyTensor groupNorm(int numGroups, LazyTensor weight, LazyTensor bias, int channelAxis, float eps) {
        return new LazyTensor(
                LazyGroupNorm.groupNorm(uop, weight.uop(), bias.uop(), numGroups, channelAxis, eps));
    }

    public LazyTensor groupNorm2d(int numGroups, LazyTensor weight, LazyTensor bias, float eps) {
        return groupNorm(numGroups, weight, bias, 1, eps);
    }

    public LazyTensor dropout(float p) {
        return dropout(p, null);
    }

    public LazyTensor dropout(float p, Long seed) {
        int[] shape = shape();
        Tensor mask = seed == null
                ? DropoutMath.sampleMask(shape, p)
                : DropoutMath.sampleMask(shape, p, seed);
        return new LazyTensor(LazyDropout.dropout(uop, LazyUOp.buffer(mask), p));
    }

    public LazyTensor dropoutWithMask(LazyTensor mask, float p) {
        return new LazyTensor(LazyDropout.dropout(uop, mask.uop(), p));
    }

    public LazyTensor maxUnpool2d(LazyTensor indices, int[] poolArg, int[] outputShape) {
        return new LazyTensor(LazyMaxUnpool2d.maxUnpool2d(uop, indices.uop(), poolArg, outputShape));
    }

    public LazyTensor sumAxis(int axis) {
        return new LazyTensor(LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, uop, new int[] {axis}));
    }

    /** Inner product / matrix multiply (tinygrad {@code dot} / {@code matmul} via broadcast mul + sum). */
    public LazyTensor dot(LazyTensor other) {
        return new LazyTensor(LazyDot.dot(uop, other.uop()));
    }

    public LazyTensor matmul(LazyTensor other) {
        return dot(other);
    }

    /** Sole leaf buffer when the DAG has exactly one BUFFER node. */
    public Tensor leafTensor() {
        List<Tensor> leaves = LazyGraph.leafBuffers(uop);
        if (leaves.size() != 1) {
            throw new IllegalStateException("expected single leaf buffer, found " + leaves.size());
        }
        return leaves.get(0);
    }

    @Override
    public String toString() {
        return "LazyTensor" + Arrays.toString(shape()) + " nodes=" + graphNodeCount()
                + (isRealized() ? " realized" : " lazy");
    }
}
