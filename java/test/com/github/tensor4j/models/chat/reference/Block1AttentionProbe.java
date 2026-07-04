/*
 * Copyright 2026 Tensor4j Maintainers
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.reference.ChatForwardTrace.Checkpoint;
import com.github.tensor4j.runtime.graph.LlamaAttentionForward;
import com.github.tensor4j.runtime.graph.LlamaBlockForward;
import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.runtime.infer.Rope;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** JSON trace of block attention sub-steps for {@code tools/compare_block1_attention.py}. */
public final class Block1AttentionProbe {

    public static void main(String[] args) throws Exception {
        int layer = envLayer();
        String path = System.getenv("TENSOR4J_GGUF_PATH");
        try (MmappedGgufFile mapped = MmappedGgufFile.open(Paths.get(path))) {
            ChatModel model = ChatModel.fromGguf(mapped);
            int[] tokens = model.tokenizer().encode("Hello");
            model.resetCache();
            List<Checkpoint> checkpoints = traceBlock(model, tokens, layer);
            System.out.print(toJson(tokens, checkpoints));
        }
    }

    private static int envLayer() {
        String raw = System.getenv("TENSOR4J_LAYER");
        if (raw == null || raw.isBlank()) {
            return 1;
        }
        return Integer.parseInt(raw.trim());
    }

    static List<Checkpoint> traceBlock1(ChatModel model, int[] tokens) {
        return traceBlock(model, tokens, 1);
    }

    static List<Checkpoint> traceBlock(ChatModel model, int[] tokens, int layer) {
        List<Checkpoint> out = new ArrayList<>();
        InferTensor x = model.embed(tokens);
        int[] positions = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            positions[i] = i;
        }
        for (int i = 0; i < layer; i++) {
            x = LlamaBlockForward.forward(
                    x,
                    model.weights().layer(i),
                    model.caches()[i],
                    model.config().nHead(),
                    model.config().nHeadKv(),
                    model.config().rmsEps(),
                    positions,
                    model.config().toRopeConfig());
        }
        if (layer > 0) {
            out.add(cpWithRow("block" + (layer - 1) + "_out", x));
        } else {
            out.add(cpWithRow("embed_out", x));
        }

        LlamaBlockForward.Weights w = model.weights().layer(layer);
        var cache = model.caches()[layer];
        var rope = model.config().toRopeConfig();
        int nHead = model.config().nHead();
        int nHeadKv = model.config().nHeadKv();
        float eps = model.config().rmsEps();
        int nEmbd = x.cols();
        int headDim = nHead > 0 ? nEmbd / nHead : 0;

        InferTensor normed = GgmlOps.rmsNorm(x, w.attnNorm().tensor(), eps);
        out.add(cp("attn_norm", normed));
        InferTensor q = GgmlOps.mulMatOut(normed, w.wq().tensor());
        InferTensor kCur = GgmlOps.mulMatOut(normed, w.wk().tensor());
        InferTensor vCur = GgmlOps.mulMatOut(normed, w.wv().tensor());
        out.add(cp("q_proj", q));
        out.add(cp("k_proj", kCur));
        out.add(cp("v_proj", vCur));

        q = Rope.applyHeads(q, nHead, headDim, positions, rope);
        kCur = Rope.applyHeads(kCur, nHeadKv, headDim, positions, rope);
        out.add(cp("q_rope_h0", headSlice(q, 0, headDim, nEmbd)));
        out.add(cp("k_rope_h0", headSlice(kCur, 0, headDim, nHeadKv * headDim)));

        cache.append(kCur, vCur);
        InferTensor[] headOut = new InferTensor[nHead];
        float scale = (float) (1.0 / Math.sqrt(headDim));
        int qHeadsPerKv = nHead / nHeadKv;
        for (int h = 0; h < nHead; h++) {
            int kvHead = h / qHeadsPerKv;
            InferTensor qHead = sliceHead(q, h, headDim, nEmbd);
            InferTensor kHead = cache.keysForHead(kvHead);
            InferTensor vHead = cache.valuesForHead(kvHead);
            InferTensor scores = GgmlOps.scale(GgmlOps.qkScores(qHead, kHead), scale);
            InferTensor probs = GgmlOps.softmaxRows(scores);
            headOut[h] = GgmlOps.attnContext(probs, vHead);
        }
        InferTensor merged = mergeHeads(headOut, headDim);
        out.add(cp("attn_merge", merged));

        InferTensor attnOut = GgmlOps.mulMatOut(merged, w.wo().tensor());
        out.add(cp("attn_wo", attnOut));
        InferTensor afterAttn = GgmlOps.add(x, attnOut);
        out.add(cp("after_attn", afterAttn));

        InferTensor ffnIn = GgmlOps.rmsNorm(afterAttn, w.ffnNorm().tensor(), eps);
        out.add(cp("ffn_norm", ffnIn));
        InferTensor gate = GgmlOps.mulMatOut(ffnIn, w.wGate().tensor());
        InferTensor up = GgmlOps.mulMatOut(ffnIn, w.wUp().tensor());
        out.add(cp("ffn_gate", gate));
        out.add(cp("ffn_up", up));
        InferTensor ffnMid = GgmlOps.swiglu(gate, up);
        out.add(cp("ffn_mid", ffnMid));
        InferTensor ffnOut = GgmlOps.mulMatOut(ffnMid, w.wDown().tensor());
        out.add(cp("ffn_out", ffnOut));
        InferTensor blockOut = GgmlOps.add(afterAttn, ffnOut);
        out.add(cp("block" + layer + "_out", blockOut));
        return out;
    }

    private static Checkpoint cpWithRow(String name, InferTensor t) {
        float[] row = lastRow(t).data();
        return new Checkpoint(name, maxAbs(row), l2(row), head(row, 8), row);
    }

    private static Checkpoint cp(String name, InferTensor t) {
        float[] row = lastRow(t).data();
        return new Checkpoint(name, maxAbs(row), l2(row), head(row, 8), null);
    }

    private static InferTensor headSlice(InferTensor q, int head, int headDim, int width) {
        return sliceHead(q, head, headDim, width);
    }

    private static InferTensor sliceHead(InferTensor q, int head, int headDim, int width) {
        float[] out = new float[q.rows() * headDim];
        float[] qd = q.data();
        for (int t = 0; t < q.rows(); t++) {
            int src = t * width + head * headDim;
            System.arraycopy(qd, src, out, t * headDim, headDim);
        }
        return InferTensor.of(out, q.rows(), headDim);
    }

    private static InferTensor mergeHeads(InferTensor[] heads, int headDim) {
        int nTokens = heads[0].rows();
        int nHead = heads.length;
        float[] out = new float[nTokens * nHead * headDim];
        for (int h = 0; h < nHead; h++) {
            float[] hd = heads[h].data();
            for (int t = 0; t < nTokens; t++) {
                int dst = t * nHead * headDim + h * headDim;
                int src = t * headDim;
                System.arraycopy(hd, src, out, dst, headDim);
            }
        }
        return InferTensor.of(out, nTokens, nHead * headDim);
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

    private static String toJson(int[] tokens, List<Checkpoint> checkpoints) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"token_ids\":").append(Arrays.toString(tokens)).append(",\"checkpoints\":[");
        for (int i = 0; i < checkpoints.size(); i++) {
            Checkpoint cp = checkpoints.get(i);
            if (i > 0) {
                sb.append(',');
            }
            sb.append("{\"name\":\"").append(cp.name()).append("\",")
                    .append("\"max_abs\":").append(cp.maxAbs()).append(',')
                    .append("\"l2\":").append(cp.l2()).append(',')
                    .append("\"head\":").append(Arrays.toString(cp.head()));
            if (cp.row() != null) {
                sb.append(",\"row\":").append(Arrays.toString(cp.row()));
            }
            sb.append('}');
        }
        sb.append("]}");
        return sb.toString();
    }
}
