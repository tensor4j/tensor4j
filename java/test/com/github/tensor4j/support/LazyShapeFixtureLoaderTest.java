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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.core.lazy.LazyShape;
import com.github.tensor4j.support.LazyShapeFixtureLoader.LazyShapeCase;
import org.junit.jupiter.api.Test;

class LazyShapeFixtureLoaderTest {

    @Test
    void parsesCommittedLazyShapeFixtures() {
        var cases = LazyShapeFixtureLoader.loadResource("/parity/tinygrad-lazy-shapes.json");
        assertEquals(6, cases.size());
        LazyShapeCase first = cases.get(0);
        assertEquals("reshape_chain", first.name());
        assertArrayEquals(new int[] {2, 3}, first.leafShape());
        assertArrayEquals(new int[] {2, 3}, LazyShapeFixtureLoader.buildLazyShape(first).shape());
    }

    @Test
    void broadcastCaseBuildsBroadcastShape() {
        LazyShapeCase parityCase = null;
        for (LazyShapeCase entry : LazyShapeFixtureLoader.loadResource("/parity/tinygrad-lazy-shapes.json")) {
            if ("broadcast_add_shapes".equals(entry.name())) {
                parityCase = entry;
                break;
            }
        }
        if (parityCase == null) {
            throw new AssertionError("broadcast_add_shapes fixture missing");
        }
        LazyShape lazy = LazyShapeFixtureLoader.buildLazyShape(parityCase);
        assertArrayEquals(new int[] {3, 4}, lazy.shape());
    }
}
