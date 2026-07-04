/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.manifold;

import java.util.ArrayList;
import java.util.List;
import com.github.tensor4j.core.Tensor;

/**
 * Euclidean parameter manifold R^n — geodesics are straight lines (tinygrad optimization landscape view).
 */
public final class EuclideanManifold {

    public ManifoldPoint point(Tensor coordinates) {
        return new ManifoldPoint(coordinates);
    }

    /** Trace gradient flow steps on the loss landscape slice. */
    public List<ManifoldPoint> gradientFlow(ManifoldPoint start, Tensor lossGradient, int steps, float stepSize) {
        List<ManifoldPoint> path = new ArrayList<>();
        ManifoldPoint current = start;
        for (int i = 0; i < steps; i++) {
            current.setTangent(lossGradient.neg());
            path.add(current);
            current = current.retract(stepSize);
        }
        return path;
    }
}
