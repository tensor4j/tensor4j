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

/** NCHW adaptive avg/max pool (tinygrad {@code adaptive_*_pool2d} subset). */
public final class AdaptivePool2dMath {

    private AdaptivePool2dMath() {
    }

    public static int[] outputShape(int[] inputNchw, int[] arg) {
        return AdaptivePool2dArg.parse(arg).outputShape(inputNchw);
    }

    public static Tensor forward(Tensor input, int[] arg) {
        AdaptivePool2dArg cfg = AdaptivePool2dArg.parse(arg);
        return Pool2dMath.forward(input, cfg.poolArg(input.shape().dims()));
    }

    public static Tensor gradInput(Tensor gradOutput, int[] arg, int[] inputShape, int[] argmaxOrNull) {
        AdaptivePool2dArg cfg = AdaptivePool2dArg.parse(arg);
        return Pool2dMath.gradInput(gradOutput, cfg.poolArg(inputShape), inputShape, argmaxOrNull);
    }
}
