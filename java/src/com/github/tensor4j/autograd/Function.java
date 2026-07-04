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

import com.github.tensor4j.core.Tensor;

/**
 * Autograd primitive. Mirrors tinygrad's gradient-flow graph at a teaching scale
 * (see micrograd / {@code Tensor.backward} in tinygrad).
 */
public abstract class Function {

    protected final Tensor[] inputs;

    protected Function(Tensor... inputs) {
        this.inputs = inputs;
    }

    public Tensor[] inputs() {
        return inputs;
    }

    public abstract Tensor forward();

    public abstract void backward(Tensor gradOutput);

    protected static void accumulate(Tensor target, Tensor contribution) {
        if (target == null || !target.requiresGrad()) {
            return;
        }
        Tensor delta = contribution.detach();
        if (target.grad() == null) {
            target.setGrad(delta);
            return;
        }
        Tensor sum = target.grad().add(delta).detach();
        target.setGrad(sum);
    }
}
