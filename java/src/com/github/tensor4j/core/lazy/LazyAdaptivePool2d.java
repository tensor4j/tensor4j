/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core.lazy;

import com.github.tensor4j.core.AdaptivePool2dArg;

/** Lazy adaptive pool via equivalent fixed {@link LazyPool2d} graph. */
final class LazyAdaptivePool2d {

    private LazyAdaptivePool2d() {
    }

    static LazyUOp adaptivePool2d(LazyUOp input, int[] arg) {
        int[] inShape = LazyUOpShapes.infer(input);
        return LazyPool2d.pool2d(input, AdaptivePool2dArg.parse(arg).poolArg(inShape));
    }
}
