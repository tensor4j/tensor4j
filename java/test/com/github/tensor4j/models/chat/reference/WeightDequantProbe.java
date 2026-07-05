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
            var header = mapped.header();
            String arch = header.findKv("general.architecture") != null
                    ? (String) header.findKv("general.architecture").value()
                    : "llama";
            boolean qwen = "qwen2".equals(arch);
            float[] norm = GgufWeightLoader.loadView(mapped, "blk.0.attn_norm.weight").toVector().data();
            System.out.println("arch=" + arch);
            System.out.println("attn_norm head=" + Arrays.toString(Arrays.copyOf(norm, 8)));
            if (qwen) {
                var wq = GgufWeightLoader.loadView(mapped, "blk.0.attn_q.weight", LlamaQkLayout.REVERSE_GGUF_DIMS, 0)
                        .toMatrix();
                System.out.println("attn_q shape=" + wq.rows() + "x" + wq.cols());
                System.out.println("attn_q row0 head=" + Arrays.toString(Arrays.copyOf(wq.data(), 16)));
                var wk = GgufWeightLoader.loadView(mapped, "blk.0.attn_k.weight", LlamaQkLayout.REVERSE_GGUF_DIMS, 0)
                        .toMatrix();
                System.out.println("attn_k shape=" + wk.rows() + "x" + wk.cols());
                System.out.println("attn_k row0 head=" + Arrays.toString(Arrays.copyOf(wk.data(), 16)));
            } else {
                float[] q = GgufWeightLoader.loadView(mapped, "blk.0.attn_q.weight", LlamaQkLayout.PERMUTE_QK, 32)
                        .toMatrix().data();
                System.out.println("attn_q row0 head=" + Arrays.toString(Arrays.copyOf(q, 8)));
            }
            float[] v = GgufWeightLoader.loadView(mapped, "blk.0.attn_v.weight", LlamaQkLayout.REVERSE_GGUF_DIMS, 0)
                    .toMatrix().data();
            System.out.println("attn_v row0 head=" + Arrays.toString(Arrays.copyOf(v, 16)));
        }
    }
}
