/*
 * Copyright 2026 Tensor4j Maintainers
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.runtime.gguf.GgufWeightLoader;
import com.github.tensor4j.runtime.gguf.LlamaQkLayout;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Paths;
import java.util.Arrays;

/** One-off weight dequant sanity check vs tinygrad. */
public final class WeightDequantProbe {
    public static void main(String[] args) throws Exception {
        String path = System.getenv("TENSOR4J_GGUF_PATH");
        try (MmappedGgufFile mapped = MmappedGgufFile.open(Paths.get(path))) {
            float[] norm = GgufWeightLoader.loadView(mapped, "blk.0.attn_norm.weight").toVector().data();
            System.out.println("attn_norm head=" + Arrays.toString(Arrays.copyOf(norm, 8)));
            float[] q = GgufWeightLoader.loadView(mapped, "blk.0.attn_q.weight", LlamaQkLayout.PERMUTE_QK, 32)
                    .toMatrix().data();
            System.out.println("attn_q row0 head=" + Arrays.toString(Arrays.copyOf(q, 8)));
            var wk = GgufWeightLoader.loadView(mapped, "blk.0.attn_k.weight", LlamaQkLayout.PERMUTE_QK_KV, 8)
                    .toMatrix();
            System.out.println("attn_k shape=" + wk.rows() + "x" + wk.cols());
            System.out.println("attn_k row0 head=" + Arrays.toString(Arrays.copyOf(wk.data(), 8)));
            System.out.println("attn_k row1 head=" + Arrays.toString(Arrays.copyOfRange(wk.data(), wk.cols(), wk.cols() + 4)));
            System.out.println("attn_k row64 head=" + Arrays.toString(Arrays.copyOfRange(wk.data(), 64 * wk.cols(), 64 * wk.cols() + 4)));
            float[] v = GgufWeightLoader.loadView(mapped, "blk.0.attn_v.weight").toMatrix().data();
            System.out.println("attn_v row0 head=" + Arrays.toString(Arrays.copyOf(v, 8)));
            float[] down = GgufWeightLoader.loadView(mapped, "blk.0.ffn_down.weight").toMatrix().data();
            System.out.println("ffn_down row0 head=" + Arrays.toString(Arrays.copyOf(down, 8)));
        }
    }
}
