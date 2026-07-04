/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.tools;

/**
 * Export tinygrad-compatible checkpoints ({@code nn.state.safe_save} / {@code get_state_dict}).
 *
 * <p>Default profile is {@code general}: build an MLP from {@code --layers} and export initialized
 * weights. Use {@code --profile algebra} for the bundled regression head.
 */
public final class GenerateWeights {

    private GenerateWeights() {
    }

    public static void main(String[] args) throws Exception {
        GenerateWeightsOptions options = GenerateWeightsOptions.parse(args);
        if (options.help()) {
            GenerateWeightsOptions.printHelp();
            return;
        }
        GenerateWeightsRunner.GenerateWeightsResult result = new GenerateWeightsRunner().run(options);
        System.out.println("wrote " + result.summary());
    }
}
