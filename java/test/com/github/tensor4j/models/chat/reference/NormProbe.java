/*
 * Copyright 2026 Tensor4j Maintainers
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import java.nio.file.Paths;
import java.util.Arrays;

/** Embed + RMS norm dump for parity debugging. */
public final class NormProbe {

    public static void main(String[] args) throws Exception {
        String path = System.getenv("TENSOR4J_GGUF_PATH");
        try (MmappedGgufFile mapped = MmappedGgufFile.open(Paths.get(path))) {
            ChatModel model = ChatModel.fromGguf(mapped);
            int[] tokens = model.tokenizer().encode("Hello");
            InferTensor x = model.embed(tokens);
            float[] row = lastRow(x).data();
            System.out.println("embed max=" + maxAbs(row) + " head=" + Arrays.toString(Arrays.copyOf(row, 8)));
            System.out.println("embed_l2=" + l2(row));
            int argmax = argMaxAbs(row);
            System.out.println("embed_argmax=" + argmax + " val=" + row[argmax]);

            var w = model.weights().layer(0);
            float eps = model.config().rmsEps();
            System.out.println("rms_eps=" + eps);
            InferTensor normed = GgmlOps.rmsNorm(x, w.attnNorm().tensor(), eps);
            float[] h = lastRow(normed).data();
            System.out.println("attn_norm max=" + maxAbs(h) + " head=" + Arrays.toString(Arrays.copyOf(h, 8)));

            // RMS without weight to isolate scale
            InferTensor scaled = GgmlOps.rmsNorm(x, ones(x.cols()), eps);
            float[] s = lastRow(scaled).data();
            System.out.println("rms_no_weight max=" + maxAbs(s) + " head=" + Arrays.toString(Arrays.copyOf(s, 8)));
        }
    }

    private static InferTensor ones(int cols) {
        float[] v = new float[cols];
        Arrays.fill(v, 1f);
        return InferTensor.vector(v);
    }

    private static InferTensor lastRow(InferTensor x) {
        int cols = x.cols();
        float[] row = new float[cols];
        System.arraycopy(x.data(), (x.rows() - 1) * cols, row, 0, cols);
        return InferTensor.vector(row);
    }

    private static int argMaxAbs(float[] row) {
        int idx = 0;
        float max = 0f;
        for (int i = 0; i < row.length; i++) {
            float a = Math.abs(row[i]);
            if (a > max) {
                max = a;
                idx = i;
            }
        }
        return idx;
    }

    private static float l2(float[] row) {
        double sum = 0.0;
        for (float v : row) {
            sum += (double) v * v;
        }
        return (float) Math.sqrt(sum);
    }

    private static float maxAbs(float[] row) {
        float max = 0f;
        for (float v : row) {
            max = Math.max(max, Math.abs(v));
        }
        return max;
    }
}
