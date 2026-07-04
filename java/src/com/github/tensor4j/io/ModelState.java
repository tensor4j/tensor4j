/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.io;

import java.util.Map;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.nn.Sequential;

/**
 * tinygrad {@code nn.state.get_state_dict} / {@code load_state_dict} for {@link Sequential}.
 * Keys follow module prefixes, e.g. {@code fc1.weight}, {@code fc1.bias}.
 */
public final class ModelState {

    private ModelState() {
    }

    public static Map<String, Tensor> getStateDict(Sequential network) {
        return ModelLoader.exportTensors(network);
    }

    public static void loadStateDict(Sequential network, Map<String, Tensor> stateDict) {
        ModelLoader.applyToSequential(network, stateDict);
    }
}
