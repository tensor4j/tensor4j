/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.gguf;

/** GGUF file constants (gguf.h). */
public final class GgufConstants {

    public static final byte[] MAGIC = new byte[] {'G', 'G', 'U', 'F'};
    public static final int VERSION = 3;
    public static final int DEFAULT_ALIGNMENT = 32;
    public static final String KEY_GENERAL_ALIGNMENT = "general.alignment";

    private GgufConstants() {
    }
}
