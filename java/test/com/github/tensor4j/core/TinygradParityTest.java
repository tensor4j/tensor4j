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

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.github.tensor4j.support.ParityFixtureLoader;
import com.github.tensor4j.support.ParityFixtureLoader.ParityCase;
import com.github.tensor4j.support.ParityFixtureLoader.ParityTensor;
import org.junit.jupiter.api.Test;

/**
 * Layer 1d: numeric parity against tinygrad-exported fixtures ({@code java/resources/parity/tinygrad-ops.json}).
 * Regenerate fixtures: {@code PYTHONPATH=vendor/tinygrad python tools/export_parity_fixtures.py}
 */
class TinygradParityTest {

    private static final String FIXTURE = "/parity/tinygrad-ops.json";

    @Test
    void opsMatchTinygradFixtures() {
        for (ParityCase parityCase : ParityFixtureLoader.loadResource(FIXTURE)) {
            if ("custom".equals(parityCase.op())) {
                continue;
            }
            Tensor actual = runCase(parityCase);
            assertNotNull(actual, "no result for case " + parityCase.name());
            assertAllClose(parityCase.expected().data(), actual, 1e-5f);
        }
    }

    private static Tensor runCase(ParityCase parityCase) {
        return switch (parityCase.op()) {
            case "layout", "custom" -> tensor(parityCase.input());
            case "reshape" -> tensor(parityCase.input()).reshape(parityCase.argShape());
            case "transpose2d" -> tensor(parityCase.input()).transpose2d();
            case "expand" -> tensor(parityCase.input()).expand(parityCase.argShape());
            case "permute" -> tensor(parityCase.input()).permute(toPermuteArgs(parityCase.argOrder()));
            case "matmul" -> tensor(parityCase.inputs().get(0)).matmul(tensor(parityCase.inputs().get(1)));
            case "relu" -> tensor(parityCase.input()).relu();
            case "add" -> tensor(parityCase.inputs().get(0)).add(tensor(parityCase.inputs().get(1)));
            case "mul" -> tensor(parityCase.inputs().get(0)).mul(tensor(parityCase.inputs().get(1)));
            case "sum" -> tensor(parityCase.input()).sum();
            case "mean" -> tensor(parityCase.input()).mean();
            case "sum_axis" -> tensor(parityCase.input()).sumAxis(parityCase.argAxis());
            default -> throw new IllegalArgumentException("unsupported parity op: " + parityCase.op());
        };
    }

    private static int[] toPermuteArgs(int[] order) {
        int[] args = order.clone();
        return args;
    }

    private static Tensor tensor(ParityTensor spec) {
        return Tensor.of(spec.data(), spec.shape());
    }
}
