/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.support;

import com.github.tensor4j.core.Tensor;

/**
 * Finite-difference gradient check (tinygrad gradcheck analogue).
 * Validates autograd against numeric estimates on the flat float buffer.
 */
public final class GradCheck {

    public static final float DEFAULT_FD_EPS = 1e-3f;
    public static final float DEFAULT_TOL = 5e-2f;

    private GradCheck() {
    }

    /** Builds a scalar loss tensor connected to {@code parameter}. */
    public interface ScalarLossGraph {
        Tensor loss(Tensor parameter);
    }

    public static void assertGradClose(Tensor parameter, ScalarLossGraph graph) {
        assertGradClose(parameter, graph, DEFAULT_FD_EPS, DEFAULT_TOL);
    }

    public static void assertGradClose(Tensor parameter, ScalarLossGraph graph, float fdEps, float tol) {
        if (!parameter.requiresGrad()) {
            throw new IllegalArgumentException("parameter must require grad");
        }
        int[] shape = parameter.shape().dims();
        float[] base = parameter.data().clone();

        parameter.zeroGrad();
        graph.loss(parameter).backward();
        if (parameter.grad() == null) {
            throw new AssertionError("analytical grad missing after backward");
        }
        float[] analytical = parameter.grad().contiguous().toFlatArray();

        float[] scratch = base.clone();
        for (int index = 0; index < base.length; index++) {
            scratch[index] = base[index] + fdEps;
            float plus = graph.loss(Tensor.of(scratch, shape).withGrad(true)).data()[0];
            scratch[index] = base[index] - fdEps;
            float minus = graph.loss(Tensor.of(scratch, shape).withGrad(true)).data()[0];
            scratch[index] = base[index];
            float numeric = (plus - minus) / (2f * fdEps);
            TensorAssert.assertClose(numeric, analytical[index], tol, "grad flat index " + index);
        }
    }
}
