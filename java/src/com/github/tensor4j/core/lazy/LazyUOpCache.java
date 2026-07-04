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

import com.github.tensor4j.core.Tensor;
import java.util.HashMap;
import java.util.Map;

/** Hash-cons cache for {@link LazyUOp} (tinygrad {@code UOpMetaClass.ucache}). */
final class LazyUOpCache {

    private static final Map<CacheKey, LazyUOp> CACHE = new HashMap<>();

    private LazyUOpCache() {
    }

    static LazyUOp intern(LazyUOp.Kind op, LazyUOp[] src, int[] arg, Tensor buffer) {
        CacheKey key = new CacheKey(op, src, arg, buffer);
        LazyUOp existing = CACHE.get(key);
        if (existing != null) {
            return existing;
        }
        LazyUOp created = new LazyUOp(op, src, arg, buffer);
        CACHE.put(key, created);
        return created;
    }

    static void clearForTests() {
        CACHE.clear();
    }

    private record CacheKey(LazyUOp.Kind op, LazyUOp[] src, int[] arg, Tensor buffer) {
        private CacheKey {
            src = src == null ? new LazyUOp[0] : src.clone();
            arg = arg == null ? null : arg.clone();
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof CacheKey cacheKey)) {
                return false;
            }
            return op == cacheKey.op && java.util.Arrays.equals(src, cacheKey.src)
                    && java.util.Arrays.equals(arg, cacheKey.arg) && buffer == cacheKey.buffer;
        }

        @Override
        public int hashCode() {
            return java.util.Arrays.hashCode(new Object[] {op, java.util.Arrays.hashCode(src),
                    java.util.Arrays.hashCode(arg), buffer});
        }
    }
}
