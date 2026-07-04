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

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.github.tensor4j.support.LazyTensorFixtureLoader;
import com.github.tensor4j.support.LazyTensorFixtureLoader.LazyTensorCase;
import org.junit.jupiter.api.Test;

/** Layer 1j: full lazy tensor chains vs tinygrad-exported expected data. */
class TinygradLazyTensorParityTest {

    private static final String FIXTURE = "/parity/tinygrad-lazy-tensor.json";

    @Test
    void lazyTensorChainsMatchFixtures() {
        for (LazyTensorCase parityCase : LazyTensorFixtureLoader.loadResource(FIXTURE)) {
            LazyTensor lazy = LazyTensorFixtureLoader.buildLazyTensor(parityCase);
            assertFalse(lazy.isRealized(), "fixture should start unrealized: " + parityCase.name());
            assertAllClose(parityCase.expected().data(), lazy.realize(), 1e-5f);
        }
    }
}
