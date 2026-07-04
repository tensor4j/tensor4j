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

import java.util.List;

/** One scheduled execution unit (tinygrad {@code CALL} / fused kernel). */
public final class LazyKernel {

    public enum Kind {
        SINGLE,
        FUSED
    }

    private final Kind kind;
    private final LazyUOp output;
    private final List<LazyUOp> body;

    private LazyKernel(Kind kind, LazyUOp output, List<LazyUOp> body) {
        this.kind = kind;
        this.output = output;
        this.body = body;
    }

    static LazyKernel single(LazyUOp node) {
        return new LazyKernel(Kind.SINGLE, node, List.of(node));
    }

    static LazyKernel fused(List<LazyUOp> orderedBody) {
        if (orderedBody.isEmpty()) {
            throw new IllegalArgumentException("fused kernel requires body");
        }
        LazyUOp sink = orderedBody.get(orderedBody.size() - 1);
        return new LazyKernel(Kind.FUSED, sink, List.copyOf(orderedBody));
    }

    public Kind kind() {
        return kind;
    }

    public LazyUOp output() {
        return output;
    }

    public List<LazyUOp> body() {
        return body;
    }

    public int opCount() {
        return body.size();
    }

    @Override
    public String toString() {
        return kind + " ops=" + opCount() + " out=" + output.op();
    }
}
