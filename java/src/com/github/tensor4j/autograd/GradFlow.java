/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.autograd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import com.github.tensor4j.core.Tensor;

/** Records gradient flow through the autograd graph (tinygrad {@code backward} trail). */
public final class GradFlow {

    private final List<String> steps = new ArrayList<>();

    public void record(String step) {
        steps.add(step);
    }

    public List<String> steps() {
        return Collections.unmodifiableList(steps);
    }

    public void clear() {
        steps.clear();
    }

    public static Tensor mseLoss(Tensor prediction, Tensor target, GradFlow flow) {
        if (flow != null) {
            flow.record("mse: (pred - target)^2 mean");
        }
        Tensor diff = prediction.sub(target);
        return diff.mul(diff).mean();
    }
}
