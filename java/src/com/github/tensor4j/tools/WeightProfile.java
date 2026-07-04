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

import java.util.Locale;

/** Weight generation profile — general state-dict export vs algebra regression bundle. */
public enum WeightProfile {

    /** Export {@code get_state_dict} for an arbitrary MLP ({@code --layers}). */
    GENERAL,

    /** Train and export the bundled high-school algebra head ({@code ax + b = c}). */
    ALGEBRA;

    public static WeightProfile fromString(String value) {
        if (value == null) {
            return GENERAL;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if ("algebra".equals(normalized) || "regression".equals(normalized)) {
            return ALGEBRA;
        }
        if ("general".equals(normalized) || "mlp".equals(normalized) || "state_dict".equals(normalized)) {
            return GENERAL;
        }
        throw new IllegalArgumentException("unknown profile: " + value + " (use general or algebra)");
    }
}
