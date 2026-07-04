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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Runs a {@link LazySchedule} (tinygrad {@code run_linear} teaching subset). */
final class LazyScheduleExecutor {

    private LazyScheduleExecutor() {
    }

    static Tensor execute(LazyUOp sink, List<LazyKernel> kernels) {
        Map<LazyUOp, Tensor> memo = new HashMap<>();
        for (LazyKernel kernel : kernels) {
            Tensor value = executeKernel(kernel, memo);
            memo.put(kernel.output(), value);
        }
        Tensor result = memo.get(sink);
        if (result == null) {
            throw new IllegalStateException("schedule did not produce sink");
        }
        return result;
    }

    private static Tensor executeKernel(LazyKernel kernel, Map<LazyUOp, Tensor> memo) {
        if (kernel.kind() == LazyKernel.Kind.FUSED) {
            return LazyKernelMath.evalFusedBody(kernel.body(), memo);
        }
        LazyUOp node = kernel.output();
        if (LazyFusion.isMovement(node.op()) || LazyFusion.isReduce(node.op())
                || node.op() == LazyUOp.Kind.RELU_MASK || node.op() == LazyUOp.Kind.WHERE
                || node.op() == LazyUOp.Kind.CONV2D || node.op() == LazyUOp.Kind.CONV2D_INPUT_GRAD
                || node.op() == LazyUOp.Kind.CONV2D_WEIGHT_GRAD || node.op() == LazyUOp.Kind.IM2COL
                || node.op() == LazyUOp.Kind.IM2COL_INPUT_GRAD || node.op() == LazyUOp.Kind.POOL2D
                || node.op() == LazyUOp.Kind.CONV_TRANSPOSE2D
                || node.op() == LazyUOp.Kind.CONV_TRANSPOSE2D_INPUT_GRAD
                || node.op() == LazyUOp.Kind.CONV_TRANSPOSE2D_WEIGHT_GRAD
                || node.op() == LazyUOp.Kind.MAX_UNPOOL2D
                || node.op() == LazyUOp.Kind.MAX_UNPOOL2D_VALUE_GRAD
                || node.op() == LazyUOp.Kind.DEQUANT_Q4_0
                || node.op() == LazyUOp.Kind.MMAP_F32) {
            return realizeWithMemo(node, memo);
        }
        if (LazyFusion.isElementwise(node.op())) {
            return LazyKernelMath.evalFusedBody(kernel.body(), memo);
        }
        throw new IllegalStateException("unsupported kernel op " + node.op());
    }

    private static Tensor realizeWithMemo(LazyUOp node, Map<LazyUOp, Tensor> memo) {
        Map<LazyUOp, Tensor> local = new HashMap<>(memo);
        return LazyRealizer.realize(node, false, local);
    }
}
