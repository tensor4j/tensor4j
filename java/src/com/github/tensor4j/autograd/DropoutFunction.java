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

import com.github.tensor4j.core.DropoutMath;
import com.github.tensor4j.core.Tensor;

public final class DropoutFunction extends Function {

    private final float p;
    private final Long seed;
    private Tensor generatedMask;

    /** Auto-mask dropout; optional {@code seed} for reproducibility. */
    public DropoutFunction(Tensor input, float p, Long seed) {
        super(input);
        this.p = p;
        this.seed = seed;
    }

    /** Dropout with a caller-supplied mask. */
    public DropoutFunction(Tensor input, Tensor mask, float p) {
        super(input, mask);
        this.p = p;
        this.seed = null;
    }

    @Override
    public Tensor forward() {
        Tensor mask = resolveMask();
        return DropoutMath.forward(inputs[0], mask, p);
    }

    @Override
    public void backward(Tensor gradOutput) {
        if (inputs[0].requiresGrad()) {
            accumulate(inputs[0], DropoutMath.gradInput(gradOutput, resolveMask(), p));
        }
    }

    private Tensor resolveMask() {
        if (inputs.length > 1) {
            return inputs[1];
        }
        if (generatedMask == null) {
            int[] shape = inputs[0].shape().dims();
            generatedMask = seed == null
                    ? DropoutMath.sampleMask(shape, p)
                    : DropoutMath.sampleMask(shape, p, seed);
        }
        return generatedMask;
    }
}
