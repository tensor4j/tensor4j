/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.runtime.graph.LlamaBlockForward;
import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import java.util.ArrayList;
import java.util.List;

/** Layer checkpoints for forward parity vs tinygrad (last-token row stats). */
public final class ChatForwardTrace {

    public record Checkpoint(String name, float maxAbs, float l2, float[] head, float[] row) {
        public Checkpoint(String name, float maxAbs, float l2, float[] head) {
            this(name, maxAbs, l2, head, null);
        }
    }

    private ChatForwardTrace() {
    }

    public static List<Checkpoint> tracePrefill(ChatModel model, int[] tokens) {
        List<Checkpoint> out = new ArrayList<>();
        InferTensor x = model.embed(tokens);
        out.add(checkpoint("embed", x));

        int past = model.kvLength();
        int[] positions = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            positions[i] = past + i;
        }

        for (int layer = 0; layer < model.config().nLayer(); layer++) {
            x = LlamaBlockForward.forward(
                    x,
                    model.weights().layer(layer),
                    model.caches()[layer],
                    model.config().nHead(),
                    model.config().nHeadKv(),
                    model.config().rmsEps(),
                    positions,
                    model.config().toRopeConfig());
            out.add(checkpoint("block_" + layer, x));
        }

        InferTensor last = lastRow(x);
        InferTensor normed = GgmlOps.rmsNorm(
                last, model.weights().outputNorm().tensor(), model.config().rmsEps());
        out.add(checkpoint("output_norm", normed));
        InferTensor logits = GgmlOps.mulMatOut(normed, model.weights().lmHead().tensor());
        out.add(checkpoint("logits", logits));
        return out;
    }

    private static Checkpoint checkpoint(String name, InferTensor tensor) {
        float[] row = lastRow(tensor).data();
        return new Checkpoint(name, maxAbs(row), l2(row), head(row, 8));
    }

    private static InferTensor lastRow(InferTensor x) {
        int cols = x.cols();
        float[] row = new float[cols];
        System.arraycopy(x.data(), (x.rows() - 1) * cols, row, 0, cols);
        return InferTensor.vector(row);
    }

    private static float maxAbs(float[] row) {
        float max = 0f;
        for (float v : row) {
            max = Math.max(max, Math.abs(v));
        }
        return max;
    }

    private static float l2(float[] row) {
        double sum = 0.0;
        for (float v : row) {
            sum += (double) v * v;
        }
        return (float) Math.sqrt(sum);
    }

    private static float[] head(float[] row, int n) {
        int len = Math.min(n, row.length);
        float[] out = new float[len];
        System.arraycopy(row, 0, out, 0, len);
        return out;
    }
}
