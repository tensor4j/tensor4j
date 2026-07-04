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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.github.tensor4j.support.LazyShapeFixtureLoader;
import com.github.tensor4j.support.LazyShapeFixtureLoader.LazyShapeCase;
import org.junit.jupiter.api.Test;

/**
 * Layer 1g: lazy shape chains match tinygrad-exported shape fixtures
 * ({@code java/resources/parity/tinygrad-lazy-shapes.json}).
 */
class TinygradLazyShapeParityTest {

    private static final String FIXTURE = "/parity/tinygrad-lazy-shapes.json";

    @Test
    void lazyShapeChainsMatchTinygradFixtures() {
        for (LazyShapeCase parityCase : LazyShapeFixtureLoader.loadResource(FIXTURE)) {
            LazyShape lazy = LazyShapeFixtureLoader.buildLazyShape(parityCase);
            assertArrayEquals(parityCase.expectedShape(), lazy.shape(),
                    "shape mismatch for case " + parityCase.name());
        }
    }
}
