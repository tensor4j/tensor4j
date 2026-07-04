/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.memory;

import com.github.tensor4j.runtime.infer.InferTensor;

/** KV cache storage for attention decode (llama-kv-cache.h). */
public interface KvCacheStore {

    int nKv();

    int maxSeq();

    void clear();

    void append(InferTensor kCur, InferTensor vCur);

    void appendBlock(InferTensor kBlock, InferTensor vBlock);

    InferTensor keysForHead(int head);

    InferTensor valuesForHead(int head);
}
