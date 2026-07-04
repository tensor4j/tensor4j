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

import com.github.tensor4j.core.Tensor;

/** Point on a differentiable manifold with optional tangent (gradient) vector. */
public final class ManifoldPoint {

    private final Tensor coordinates;
    private Tensor tangent;

    public ManifoldPoint(Tensor coordinates) {
        this.coordinates = coordinates;
    }

    public Tensor coordinates() {
        return coordinates;
    }

    public Tensor tangent() {
        return tangent;
    }

    public void setTangent(Tensor tangent) {
        this.tangent = tangent;
    }

    public ManifoldPoint retract(float step) {
        if (tangent == null) {
            return this;
        }
        Tensor moved = coordinates.add(tangent.mul(Tensor.of(step)));
        return new ManifoldPoint(moved);
    }
}
