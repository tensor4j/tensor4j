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

public final class NegFunction extends Function {

    public NegFunction(Tensor input) {
        super(input);
    }

    @Override
    public Tensor forward() {
        return inputs[0].neg();
    }

    @Override
    public void backward(Tensor gradOutput) {
        accumulate(inputs[0], gradOutput.neg());
    }
}
