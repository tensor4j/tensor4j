/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.nn;

import java.util.List;
import com.github.tensor4j.core.Tensor;

/** Base module (tinygrad {@code nn.Module}). */
public abstract class Module {

    public abstract Tensor forward(Tensor input);

    public List<Tensor> parameters() {
        return List.of();
    }

    public void zeroGrad() {
        for (Tensor param : parameters()) {
            param.zeroGrad();
        }
    }
}
