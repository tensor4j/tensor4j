/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.core;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

/**
 * GPU tensor with autograd support. Wraps {@link GpuTensor} and records
 * a computation graph for automatic differentiation.
 *
 * <p>Usage:
 * <pre>{@code
 * GpuAutogradTensor a = GpuAutogradTensor.fromHost(dev, new float[]{1,2,3}, 3).requiresGrad(true);
 * GpuAutogradTensor b = GpuAutogradTensor.fromHost(dev, new float[]{4,5,6}, 3).requiresGrad(true);
 * GpuAutogradTensor c = a.add(b).mul(b);
 * c.backward();
 * System.out.println(a.grad().toHost()[0]); // ∂c/∂a[0] = b[0] = 4
 * }</pre>
 */
public final class GpuAutogradTensor implements AutoCloseable {

    private GpuTensor data;
    private final GpuDevice device;
    private boolean requiresGrad;
    private GpuTensor grad;
    private GradFn gradFn;
    private final List<GpuAutogradTensor> children;

    public GpuAutogradTensor(GpuTensor data, GpuDevice device) {
        this.data = data;
        this.device = device;
        this.children = new ArrayList<>();
    }

    public static GpuAutogradTensor fromHost(GpuDevice device, float[] host, int... shape) {
        return new GpuAutogradTensor(GpuTensor.fromHost(device, host, shape), device);
    }

    public GpuAutogradTensor requiresGrad(boolean v) {
        this.requiresGrad = v;
        return this;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public GpuTensor grad() {
        return grad;
    }

    public GpuTensor data() {
        return data;
    }

    public GpuDevice device() {
        return device;
    }

    public void zeroGrad() {
        if (grad != null) {
            grad.close();
            grad = null;
        }
    }

    public void setData(GpuTensor newData) {
        if (this.data.buffer() != newData.buffer()) {
            this.data.close();
        }
        this.data = newData;
    }

    public int[] shape() {
        return data.shape();
    }

    public int numel() {
        return data.numel();
    }

    public float[] toHost() {
        return data.toHost();
    }

    @Override
    public void close() {
        data.close();
    }

    public void backward() {
        if (!requiresGrad && gradFn == null) {
            return;
        }
        if (grad == null) {
            int numel = data.numel();
            GpuBuffer gBuf = device.allocate((long) numel * 4L);
            float[] ones = new float[numel];
            Arrays.fill(ones, 1f);
            gBuf.copyFromHost(ones, 0, 0, numel);
            grad = new GpuTensor(gBuf, data.shape(), data.strides(), 0);
        }
        markAllUnvisited(this);
        Deque<GpuAutogradTensor> topo = new ArrayDeque<>();
        buildTopo(this, topo);
        for (GpuAutogradTensor t : topo) {
            if (t.gradFn != null) {
                t.gradFn.apply(t.grad);
            }
        }
    }

    public void accumulateGrad(GpuTensor g) {
        if (!requiresGrad) return;
        if (grad == null) {
            grad = g;
            return;
        }
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor s = math.add(grad, g);
        grad.close();
        grad = s;
    }

    private void propagateGrad(GpuTensor g) {
        if (requiresGrad) {
            accumulateGrad(g);
        } else if (gradFn != null) {
            if (grad == null) {
                float[] data = g.toHost();
                GpuBuffer buf = device.allocate((long) data.length * 4L);
                buf.copyFromHost(data, 0, 0, data.length);
                grad = new GpuTensor(buf, g.shape(), computeStrides(g.shape()), 0);
            } else {
                GpuTensorMath math = new GpuTensorMath(device);
                GpuTensor s = math.add(grad, g);
                grad.close();
                grad = s;
            }
        }
    }

    private static void buildTopo(GpuAutogradTensor t, Deque<GpuAutogradTensor> topo) {
        if (!t.requiresGrad && t.gradFn == null) return;
        if (t.visited) return;
        t.visited = true;
        if (t.gradFn != null) {
            for (GpuAutogradTensor child : t.children) {
                buildTopo(child, topo);
            }
        }
        topo.addFirst(t);
    }

    private boolean visited;

    private void markUnvisited() {
        visited = false;
    }

    static void markAllUnvisited(GpuAutogradTensor t) {
        Deque<GpuAutogradTensor> stack = new ArrayDeque<>();
        stack.push(t);
        while (!stack.isEmpty()) {
            GpuAutogradTensor cur = stack.pop();
            cur.markUnvisited();
            for (GpuAutogradTensor child : cur.children) {
                stack.push(child);
            }
        }
    }

    public GpuAutogradTensor add(GpuAutogradTensor other) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.add(this.data, other.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if ((this.requiresGrad || this.gradFn != null) || (other.requiresGrad || other.gradFn != null)) {
            out.gradFn = (gradOutput) -> {
                this.propagateGrad(gradOutput);
                other.propagateGrad(gradOutput);
            };
            captureChildren(out, this, other);
        }
        return out;
    }

    public GpuAutogradTensor sub(GpuAutogradTensor other) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.sub(this.data, other.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if ((this.requiresGrad || this.gradFn != null) || (other.requiresGrad || other.gradFn != null)) {
            out.gradFn = (gradOutput) -> {
                this.propagateGrad(gradOutput);
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor ng = m.neg(gradOutput);
                other.propagateGrad(ng);
            };
            captureChildren(out, this, other);
        }
        return out;
    }

    public GpuAutogradTensor mul(GpuAutogradTensor other) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.mul(this.data, other.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if ((this.requiresGrad || this.gradFn != null) || (other.requiresGrad || other.gradFn != null)) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                this.propagateGrad(m.mul(gradOutput, other.data));
                other.propagateGrad(m.mul(gradOutput, this.data));
            };
            captureChildren(out, this, other);
        }
        return out;
    }

    public GpuAutogradTensor div(GpuAutogradTensor other) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.div(this.data, other.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if ((this.requiresGrad || this.gradFn != null) || (other.requiresGrad || other.gradFn != null)) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                if (this.requiresGrad || this.gradFn != null) {
                    int[] shape = other.data.shape();
                    int numel = other.data.numel();
                    float[] ones = new float[numel];
                    Arrays.fill(ones, 1f);
                    GpuBuffer onesBuf = device.allocate((long) numel * 4L);
                    onesBuf.copyFromHost(ones, 0, 0, numel);
                    GpuTensor onesT = new GpuTensor(onesBuf, shape,
                        computeStrides(shape), 0);
                    GpuTensor invB = m.div(onesT, other.data);
                    GpuTensor gA = m.mul(gradOutput, invB);
                    this.propagateGrad(gA);
                }
                if (other.requiresGrad || other.gradFn != null) {
                    GpuTensor aDivBSq = m.div(r, other.data);
                    GpuTensor ng = m.neg(aDivBSq);
                    GpuTensor g = m.mul(gradOutput, ng);
                    other.propagateGrad(g);
                }
            };
            captureChildren(out, this, other);
        }
        return out;
    }

    public GpuAutogradTensor relu() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.relu(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                float[] ha = this.data.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[ha.length];
                for (int i = 0; i < ha.length; i++) {
                    hc[i] = ha[i] > 0f ? hg[i] : 0f;
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape,
                    computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor neg() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.neg(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor ng = m.neg(gradOutput);
                this.propagateGrad(ng);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor scale(float s) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.scale(this.data, s);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor g = m.scale(gradOutput, s);
                this.propagateGrad(g);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor exp() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.exp(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor g = m.mul(gradOutput, r);
                this.propagateGrad(g);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor log() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.log(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                int[] shape = this.data.shape();
                int numel = this.data.numel();
                float[] ones = new float[numel];
                Arrays.fill(ones, 1f);
                GpuBuffer onesBuf = device.allocate((long) numel * 4L);
                onesBuf.copyFromHost(ones, 0, 0, numel);
                GpuTensor onesT = new GpuTensor(onesBuf, shape,
                    computeStrides(shape), 0);
                GpuTensor invA = m.div(onesT, this.data);
                GpuTensor g = m.mul(gradOutput, invA);
                this.propagateGrad(g);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor sqrt() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.sqrt(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                int[] shape = this.data.shape();
                int numel = this.data.numel();
                float[] ones = new float[numel];
                Arrays.fill(ones, 1f);
                GpuBuffer onesBuf = device.allocate((long) numel * 4L);
                onesBuf.copyFromHost(ones, 0, 0, numel);
                GpuTensor onesT = new GpuTensor(onesBuf, shape,
                    computeStrides(shape), 0);
                GpuTensor inv = m.div(onesT, r);
                GpuTensor half = m.scale(inv, 0.5f);
                GpuTensor g = m.mul(gradOutput, half);
                this.propagateGrad(g);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor sigmoid() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.sigmoid(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                float[] hr = r.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[hr.length];
                for (int i = 0; i < hr.length; i++) {
                    hc[i] = hg[i] * hr[i] * (1f - hr[i]);
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape,
                    computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor tanh() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.tanh(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                float[] hr = r.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[hr.length];
                for (int i = 0; i < hr.length; i++) {
                    hc[i] = hg[i] * (1f - hr[i] * hr[i]);
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape,
                    computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor matmul(GpuAutogradTensor other) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.matmul(this.data, other.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if ((this.requiresGrad || this.gradFn != null) || (other.requiresGrad || other.gradFn != null)) {
            int[] sa = this.data.shape();
            int[] sb = other.data.shape();
            int m = sa[0];
            int k = sa[1];
            int n = sb[1];
            out.gradFn = (gradOutput) -> {
                GpuTensorMath mth = new GpuTensorMath(device);
                if (this.requiresGrad || this.gradFn != null) {
                    GpuTensor bT = transposeCopy(other.data, k, n);
                    GpuTensor g = mth.matmul(gradOutput, bT);
                    this.propagateGrad(g);
                }
                if (other.requiresGrad || other.gradFn != null) {
                    GpuTensor aT = transposeCopy(this.data, m, k);
                    GpuTensor g = mth.matmul(aT, gradOutput);
                    other.propagateGrad(g);
                }
            };
            captureChildren(out, this, other);
        }
        return out;
    }

    public GpuAutogradTensor squaredDifference(GpuAutogradTensor other) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.squaredDifference(this.data, other.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if ((this.requiresGrad || this.gradFn != null) || (other.requiresGrad || other.gradFn != null)) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor diff = m.sub(this.data, other.data);
                GpuTensor twoDiff = m.scale(diff, 2f);
                this.propagateGrad(m.mul(gradOutput, twoDiff));
                GpuTensor ng = m.neg(twoDiff);
                other.propagateGrad(m.mul(gradOutput, ng));
            };
            captureChildren(out, this, other);
        }
        return out;
    }

    public GpuAutogradTensor sum() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.sum(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                float gVal = gradOutput.toHost()[0];
                int n = this.data.numel();
                float[] hc = new float[n];
                Arrays.fill(hc, gVal);
                GpuBuffer buf = device.allocate((long) n * 4L);
                buf.copyFromHost(hc, 0, 0, n);
                GpuTensor gg = new GpuTensor(buf, this.data.shape(),
                    this.data.strides(), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor pow(GpuAutogradTensor exponent) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.pow(this.data, exponent.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if ((this.requiresGrad || this.gradFn != null) || (exponent.requiresGrad || exponent.gradFn != null)) {
            int numel = exponent.data.numel();
            float[] onesArr = new float[numel];
            Arrays.fill(onesArr, 1f);
            GpuBuffer onesBuf = device.allocate((long) numel * 4L);
            onesBuf.copyFromHost(onesArr, 0, 0, numel);
            GpuTensor onesT = new GpuTensor(onesBuf, exponent.data.shape(),
                computeStrides(exponent.data.shape()), 0);
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                if (this.requiresGrad || this.gradFn != null) {
                    // dz/dx = y * x^(y-1)
                    GpuTensor expMinus1 = m.sub(exponent.data, onesT);
                    GpuTensor basePow = m.pow(this.data, expMinus1);
                    GpuTensor yBase = m.mul(exponent.data, basePow);
                    GpuTensor g = m.mul(gradOutput, yBase);
                    this.propagateGrad(g);
                }
                if (exponent.requiresGrad || exponent.gradFn != null) {
                    // dz/dy = x^y * ln(x) = r * ln(x)
                    GpuTensor logX = m.log(this.data);
                    GpuTensor g = m.mul(gradOutput, m.mul(r, logX));
                    exponent.propagateGrad(g);
                }
            };
            captureChildren(out, this, exponent);
        }
        return out;
    }

    public GpuAutogradTensor mean() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.mean(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            float invN = 1f / this.data.numel();
            out.gradFn = (gradOutput) -> {
                float gVal = gradOutput.toHost()[0] * invN;
                int n = this.data.numel();
                float[] hc = new float[n];
                Arrays.fill(hc, gVal);
                GpuBuffer buf = device.allocate((long) n * 4L);
                buf.copyFromHost(hc, 0, 0, n);
                GpuTensor gg = new GpuTensor(buf, this.data.shape(),
                    this.data.strides(), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor abs() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.abs(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                float[] ha = this.data.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[ha.length];
                for (int i = 0; i < ha.length; i++) {
                    hc[i] = ha[i] > 0f ? hg[i] : (ha[i] < 0f ? -hg[i] : 0f);
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape,
                    computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor clip(float minVal, float maxVal) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.clip(this.data, minVal, maxVal);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                float[] ha = this.data.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[ha.length];
                for (int i = 0; i < ha.length; i++) {
                    hc[i] = (ha[i] >= minVal && ha[i] <= maxVal) ? hg[i] : 0f;
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape,
                    computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor transpose() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.transpose(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor g = m.transpose(gradOutput);
                this.propagateGrad(g);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor sin() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.sin(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor cosX = m.cos(this.data);
                GpuTensor g = m.mul(gradOutput, cosX);
                this.propagateGrad(g);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor cos() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.cos(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor sinX = m.sin(this.data);
                GpuTensor ng = m.neg(sinX);
                GpuTensor g = m.mul(gradOutput, ng);
                this.propagateGrad(g);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor reshape(int... newShape) {
        GpuTensor r = this.data.reshape(newShape);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] origShape = this.data.shape();
            out.gradFn = (gradOutput) -> {
                GpuTensor gg = new GpuTensor(gradOutput.buffer(), origShape,
                    GpuTensor.computeStrides(origShape), gradOutput.offset());
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor squeeze() {
        GpuTensor r = this.data.squeeze();
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] origShape = this.data.shape();
            out.gradFn = (gradOutput) -> {
                GpuTensor gg = new GpuTensor(gradOutput.buffer(), origShape,
                    GpuTensor.computeStrides(origShape), gradOutput.offset());
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor unsqueeze(int dim) {
        GpuTensor r = this.data.unsqueeze(dim);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] origShape = this.data.shape();
            out.gradFn = (gradOutput) -> {
                GpuTensor gg = new GpuTensor(gradOutput.buffer(), origShape,
                    GpuTensor.computeStrides(origShape), gradOutput.offset());
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor gelu() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.gelu(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                float[] ha = this.data.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[ha.length];
                float sqrt2 = 1.41421356f;
                float sqrt2Pi = 2.50662827f; // sqrt(2*pi)
                for (int i = 0; i < ha.length; i++) {
                    float x = ha[i];
                    float t = x / sqrt2;
                    float erf = (float) Math.signum(t) *
                        (1f - (float) Math.exp(-t * t * (
                            1.2732416f + 0.147f * t * t)));
                    // crude erf approximation
                    hc[i] = hg[i] * (
                        0.5f * (1f + erf) +
                        x * (float) Math.exp(-x * x * 0.5f) / sqrt2Pi
                    );
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape,
                    computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor leakyRelu(float alpha) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.leakyRelu(this.data, alpha);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            out.gradFn = (gradOutput) -> {
                float[] ha = this.data.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[ha.length];
                for (int i = 0; i < ha.length; i++) {
                    hc[i] = ha[i] > 0f ? hg[i] : hg[i] * alpha;
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape,
                    computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor dropout(float prob, java.util.Random rng) {
        GpuTensorMath math = new GpuTensorMath(device);
        int n = this.data.numel();
        float[] mask = new float[n];
        for (int i = 0; i < n; i++) {
            mask[i] = rng.nextFloat();
        }
        GpuTensor maskT = GpuTensor.fromHost(device, mask, this.data.shape());
        GpuTensor r = math.dropout(this.data, prob, rng);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            float pClamped = Math.max(0f, Math.min(1f, prob));
            float pFinal = pClamped;
            float scale = pFinal >= 1f ? 1f : 1f / (1f - pFinal);
            out.gradFn = (gradOutput) -> {
                float[] hMask = maskT.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[hg.length];
                for (int i = 0; i < hg.length; i++) {
                    hc[i] = hMask[i] < pFinal ? 0f : hg[i] * scale;
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape,
                    computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor binaryCrossEntropy(GpuAutogradTensor target) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.binaryCrossEntropy(this.data, target.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int n = this.data.numel();
            int[] shape = this.data.shape();
            float[] hPred = this.data.toHost();
            float[] hOne = new float[n];
            float[] hClip = new float[n];
            for (int i = 0; i < n; i++) {
                hClip[i] = Math.max(Math.min(hPred[i], 0.9999999f), 1e-7f);
                hOne[i] = 1f;
            }
            GpuBuffer oneBuf = device.allocate((long) n * 4L);
            oneBuf.copyFromHost(hOne, 0, 0, n);
            GpuTensor ones = new GpuTensor(oneBuf, shape,
                computeStrides(shape), 0);
            GpuBuffer clipBuf = device.allocate((long) n * 4L);
            clipBuf.copyFromHost(hClip, 0, 0, n);
            GpuTensor clipped = new GpuTensor(clipBuf, shape,
                computeStrides(shape), 0);
            out.gradFn = (gradOutput) -> {
                float gVal = gradOutput.toHost()[0];
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor oneMinusClip = m.sub(ones, clipped);
                GpuTensor denom = m.mul(clipped, oneMinusClip);
                GpuTensor diff = m.sub(clipped, target.data);
                GpuTensor g = m.div(diff, denom);
                if (gVal != 1f) {
                    g = m.scale(g, gVal);
                }
                this.propagateGrad(g);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor silu() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.silu(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            float[] hs = r.toHost();
            float[] hx = this.data.toHost();
            out.gradFn = (gradOutput) -> {
                float[] hg = gradOutput.toHost();
                float[] hc = new float[hg.length];
                for (int i = 0; i < hg.length; i++) {
                    float sigmoid = 1f / (1f + (float)Math.exp(-hx[i]));
                    hc[i] = hg[i] * (hs[i] + sigmoid * (1f - hs[i]));
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape, computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor elu(float alpha) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.elu(this.data, alpha);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            float[] hx = this.data.toHost();
            out.gradFn = (gradOutput) -> {
                float[] hg = gradOutput.toHost();
                float[] hc = new float[hg.length];
                for (int i = 0; i < hg.length; i++) {
                    hc[i] = hx[i] > 0f ? hg[i] : hg[i] * alpha * (float)Math.exp(hx[i]);
                }
                int[] shape = gradOutput.shape();
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape, computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor flatten(int startDim) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.flatten(this.data, startDim);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] origShape = this.data.shape();
            out.gradFn = (gradOutput) -> {
                float[] hg = gradOutput.toHost();
                GpuBuffer buf = device.allocate((long) hg.length * 4L);
                buf.copyFromHost(hg, 0, 0, hg.length);
                GpuTensor gg = new GpuTensor(buf, origShape,
                    computeStrides(origShape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor embedding(GpuAutogradTensor indices) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.embedding(this.data, indices.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] ws = this.data.shape();
            int numEmbed = ws[0];
            int dim = ws[1];
            float[] hIdx = indices.data.toHost();
            int n = hIdx.length;
            out.gradFn = (gradOutput) -> {
                float[] hg = gradOutput.toHost();
                float[] dw = new float[numEmbed * dim];
                for (int i = 0; i < n; i++) {
                    int idx = (int) hIdx[i];
                    for (int j = 0; j < dim; j++) {
                        dw[idx * dim + j] += hg[i * dim + j];
                    }
                }
                GpuBuffer buf = device.allocate((long) dw.length * 4L);
                buf.copyFromHost(dw, 0, 0, dw.length);
                this.propagateGrad(new GpuTensor(buf, ws, computeStrides(ws), 0));
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor conv2d(GpuAutogradTensor kernel, int stride, int padding) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.conv2d(this.data, kernel.data, stride, padding);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        boolean needsGrad = (this.requiresGrad || this.gradFn != null)
            || (kernel.requiresGrad || kernel.gradFn != null);
        if (needsGrad) {
            int[] is = this.data.shape();
            int[] ks = kernel.data.shape();
            int n = is[0], inC = is[1], h = is[2], w = is[3];
            int outC = ks[0], kH = ks[2], kW = ks[3];
            int outH = (h + 2 * padding - kH) / stride + 1;
            int outW = (w + 2 * padding - kW) / stride + 1;
            float[] hi = this.data.toHost();
            float[] hk = kernel.data.toHost();
            out.gradFn = (gradOutput) -> {
                float[] go = gradOutput.toHost();
                GpuTensorMath m = new GpuTensorMath(device);
                if (this.requiresGrad || this.gradFn != null) {
                    float[] dx = new float[n * inC * h * w];
                    for (int ni = 0; ni < n; ni++) {
                        for (int oi = 0; oi < outC; oi++) {
                            for (int oh = 0; oh < outH; oh++) {
                                for (int ow = 0; ow < outW; ow++) {
                                    float g = go[((ni * outC + oi) * outH + oh) * outW + ow];
                                    for (int ci = 0; ci < inC; ci++) {
                                        for (int kh = 0; kh < kH; kh++) {
                                            for (int kw = 0; kw < kW; kw++) {
                                                int ih = oh * stride - padding + kh;
                                                int iw = ow * stride - padding + kw;
                                                if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                                    int inIdx = ((ni * inC + ci) * h + ih) * w + iw;
                                                    int kIdx = ((oi * inC + ci) * kH + kh) * kW + kw;
                                                    dx[inIdx] += g * hk[kIdx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    GpuBuffer dxBuf = device.allocate((long) dx.length * 4L);
                    dxBuf.copyFromHost(dx, 0, 0, dx.length);
                    this.propagateGrad(new GpuTensor(dxBuf, is, computeStrides(is), 0));
                }
                if (kernel.requiresGrad || kernel.gradFn != null) {
                    float[] dk = new float[outC * inC * kH * kW];
                    for (int ni = 0; ni < n; ni++) {
                        for (int oi = 0; oi < outC; oi++) {
                            for (int oh = 0; oh < outH; oh++) {
                                for (int ow = 0; ow < outW; ow++) {
                                    float g = go[((ni * outC + oi) * outH + oh) * outW + ow];
                                    for (int ci = 0; ci < inC; ci++) {
                                        for (int kh = 0; kh < kH; kh++) {
                                            for (int kw = 0; kw < kW; kw++) {
                                                int ih = oh * stride - padding + kh;
                                                int iw = ow * stride - padding + kw;
                                                if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                                    int inIdx = ((ni * inC + ci) * h + ih) * w + iw;
                                                    int kIdx = ((oi * inC + ci) * kH + kh) * kW + kw;
                                                    dk[kIdx] += g * hi[inIdx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    GpuBuffer dkBuf = device.allocate((long) dk.length * 4L);
                    dkBuf.copyFromHost(dk, 0, 0, dk.length);
                    kernel.propagateGrad(new GpuTensor(dkBuf, ks, computeStrides(ks), 0));
                }
            };
            if (kernel != this) captureChildren(out, this, kernel);
            else captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor maxPool2d(int kernelSize, int stride, int padding) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.maxPool2d(this.data, kernelSize, stride, padding);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] is = this.data.shape();
            int n = is[0], c = is[1], h = is[2], w = is[3];
            int outH = (h + 2 * padding - kernelSize) / stride + 1;
            int outW = (w + 2 * padding - kernelSize) / stride + 1;
            float[] hi = this.data.toHost();
            out.gradFn = (gradOutput) -> {
                float[] go = gradOutput.toHost();
                float[] dx = new float[n * c * h * w];
                for (int ni = 0; ni < n; ni++) {
                    for (int ci = 0; ci < c; ci++) {
                        for (int oh = 0; oh < outH; oh++) {
                            for (int ow = 0; ow < outW; ow++) {
                                float g = go[((ni * c + ci) * outH + oh) * outW + ow];
                                float maxVal = Float.NEGATIVE_INFINITY;
                                int maxIdx = -1;
                                for (int kh = 0; kh < kernelSize; kh++) {
                                    for (int kw = 0; kw < kernelSize; kw++) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                            int inIdx = ((ni * c + ci) * h + ih) * w + iw;
                                            if (hi[inIdx] > maxVal) {
                                                maxVal = hi[inIdx];
                                                maxIdx = inIdx;
                                            }
                                        }
                                    }
                                }
                                if (maxIdx >= 0) dx[maxIdx] += g;
                            }
                        }
                    }
                }
                GpuBuffer dxBuf = device.allocate((long) dx.length * 4L);
                dxBuf.copyFromHost(dx, 0, 0, dx.length);
                this.propagateGrad(new GpuTensor(dxBuf, is, computeStrides(is), 0));
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor mseLoss(GpuAutogradTensor target) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.mseLoss(this.data, target.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int n = this.data.numel();
            int[] shape = this.data.shape();
            float[] hPred = this.data.toHost();
            float[] hTarget = target.data.toHost();
            out.gradFn = (gradOutput) -> {
                float gVal = gradOutput.toHost()[0];
                float[] hg = new float[n];
                for (int i = 0; i < n; i++) {
                    hg[i] = 2f * (hPred[i] - hTarget[i]) * gVal;
                }
                GpuBuffer buf = device.allocate((long) n * 4L);
                buf.copyFromHost(hg, 0, 0, n);
                GpuTensor gg = new GpuTensor(buf, shape, computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor softmax() {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.softmax(this.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] shape = this.data.shape();
            out.gradFn = (gradOutput) -> {
                GpuTensorMath m = new GpuTensorMath(device);
                float[] hy = r.toHost();
                float[] hg = gradOutput.toHost();
                float[] hc = new float[hy.length];
                if (shape.length == 1) {
                    float dot = 0f;
                    for (int i = 0; i < hy.length; i++) dot += hg[i] * hy[i];
                    for (int i = 0; i < hy.length; i++) hc[i] = hy[i] * (hg[i] - dot);
                } else {
                    int rows = shape[0];
                    int cols = shape[1];
                    for (int row = 0; row < rows; row++) {
                        int base = row * cols;
                        float dot = 0f;
                        for (int j = 0; j < cols; j++) dot += hg[base + j] * hy[base + j];
                        for (int j = 0; j < cols; j++) hc[base + j] = hy[base + j] * (hg[base + j] - dot);
                    }
                }
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, shape, computeStrides(shape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor crossEntropy(GpuAutogradTensor targets) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.crossEntropy(this.data, targets.data);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] ls = this.data.shape();
            int n = ls[0];
            int c = ls[1];
            float[] hTargets = targets.data.toHost();
            out.gradFn = (gradOutput) -> {
                float gVal = gradOutput.toHost()[0];
                GpuTensorMath m = new GpuTensorMath(device);
                GpuTensor probs = m.softmax2d(this.data);
                float[] hp = probs.toHost();
                float[] hg = new float[n * c];
                for (int i = 0; i < n; i++) {
                    int t = (int) hTargets[i];
                    for (int j = 0; j < c; j++) {
                        hg[i * c + j] = hp[i * c + j];
                    }
                    hg[i * c + t] -= 1f;
                }
                for (int i = 0; i < hg.length; i++) hg[i] *= gVal;
                GpuBuffer buf = device.allocate((long) hg.length * 4L);
                buf.copyFromHost(hg, 0, 0, hg.length);
                GpuTensor gg = new GpuTensor(buf, ls, computeStrides(ls), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor layerNorm(GpuAutogradTensor gamma, GpuAutogradTensor beta, float eps) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.layerNorm(this.data, gamma.data, beta.data, eps);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        boolean needsGrad = (this.requiresGrad || this.gradFn != null)
            || (gamma.requiresGrad || gamma.gradFn != null)
            || (beta.requiresGrad || beta.gradFn != null);
        if (needsGrad) {
            int[] xs = this.data.shape();
            int b = xs[0];
            int d = xs[1];
            float[] hx = this.data.toHost();
            float[] hg = gamma.data.toHost();
            float[] hb = beta.data.toHost();
            float[] hMean = new float[b];
            float[] hInvStd = new float[b];
            float[] hNorm = new float[b * d];
            for (int i = 0; i < b; i++) {
                int base = i * d;
                float mean = 0f;
                for (int j = 0; j < d; j++) mean += hx[base + j];
                mean /= d;
                hMean[i] = mean;
                float var = 0f;
                for (int j = 0; j < d; j++) {
                    float diff = hx[base + j] - mean;
                    var += diff * diff;
                }
                var /= d;
                hInvStd[i] = 1f / (float) Math.sqrt(var + eps);
                for (int j = 0; j < d; j++) {
                    hNorm[base + j] = (hx[base + j] - mean) * hInvStd[i];
                }
            }
            out.gradFn = (gradOutput) -> {
                float[] go = gradOutput.toHost();
                if (this.requiresGrad || this.gradFn != null) {
                    float[] dx = new float[b * d];
                    for (int i = 0; i < b; i++) {
                        int base = i * d;
                        float dYsum = 0f, dYxNormSum = 0f;
                        for (int j = 0; j < d; j++) {
                            dYsum += go[base + j];
                            dYxNormSum += go[base + j] * hNorm[base + j];
                        }
                        for (int j = 0; j < d; j++) {
                            dx[base + j] = hg[j] * hInvStd[i] / d * (
                                d * go[base + j] - dYsum - hNorm[base + j] * dYxNormSum);
                        }
                    }
                    GpuBuffer dxBuf = device.allocate((long) dx.length * 4L);
                    dxBuf.copyFromHost(dx, 0, 0, dx.length);
                    this.propagateGrad(new GpuTensor(dxBuf, xs, computeStrides(xs), 0));
                }
                if (gamma.requiresGrad || gamma.gradFn != null) {
                    float[] dg = new float[d];
                    for (int i = 0; i < b; i++) {
                        int base = i * d;
                        for (int j = 0; j < d; j++) {
                            dg[j] += go[base + j] * hNorm[base + j];
                        }
                    }
                    GpuBuffer dgBuf = device.allocate((long) dg.length * 4L);
                    dgBuf.copyFromHost(dg, 0, 0, dg.length);
                    gamma.propagateGrad(new GpuTensor(dgBuf, gamma.data.shape(),
                        computeStrides(gamma.data.shape()), 0));
                }
                if (beta.requiresGrad || beta.gradFn != null) {
                    float[] db = new float[d];
                    for (int i = 0; i < b; i++) {
                        int base = i * d;
                        for (int j = 0; j < d; j++) {
                            db[j] += go[base + j];
                        }
                    }
                    GpuBuffer dbBuf = device.allocate((long) db.length * 4L);
                    dbBuf.copyFromHost(db, 0, 0, db.length);
                    beta.propagateGrad(new GpuTensor(dbBuf, beta.data.shape(),
                        computeStrides(beta.data.shape()), 0));
                }
            };
            if (gamma != this) captureChildren(out, this, gamma);
            else captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor batchNorm1d(GpuAutogradTensor gamma, GpuAutogradTensor beta, float eps) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.batchNorm1d(this.data, gamma.data, beta.data, eps);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        boolean needsGrad = (this.requiresGrad || this.gradFn != null)
            || (gamma.requiresGrad || gamma.gradFn != null)
            || (beta.requiresGrad || beta.gradFn != null);
        if (needsGrad) {
            int[] xs = this.data.shape();
            int b = xs[0];
            int d = xs[1];
            float[] hx = this.data.toHost();
            float[] hg = gamma.data.toHost();
            float[] hInvStd = new float[d];
            float[] hNorm = new float[b * d];
            for (int j = 0; j < d; j++) {
                float mean = 0f;
                for (int i = 0; i < b; i++) mean += hx[i * d + j];
                mean /= b;
                float var = 0f;
                for (int i = 0; i < b; i++) {
                    float diff = hx[i * d + j] - mean;
                    var += diff * diff;
                }
                var /= b;
                hInvStd[j] = 1f / (float) Math.sqrt(var + eps);
                for (int i = 0; i < b; i++) {
                    hNorm[i * d + j] = (hx[i * d + j] - mean) * hInvStd[j];
                }
            }
            out.gradFn = (gradOutput) -> {
                float[] go = gradOutput.toHost();
                if (this.requiresGrad || this.gradFn != null) {
                    float[] dx = new float[b * d];
                    for (int j = 0; j < d; j++) {
                        float dYsum = 0f, dYxNormSum = 0f;
                        for (int i = 0; i < b; i++) {
                            dYsum += go[i * d + j];
                            dYxNormSum += go[i * d + j] * hNorm[i * d + j];
                        }
                        for (int i = 0; i < b; i++) {
                            dx[i * d + j] = hg[j] * hInvStd[j] / b * (
                                b * go[i * d + j] - dYsum - hNorm[i * d + j] * dYxNormSum);
                        }
                    }
                    GpuBuffer dxBuf = device.allocate((long) dx.length * 4L);
                    dxBuf.copyFromHost(dx, 0, 0, dx.length);
                    this.propagateGrad(new GpuTensor(dxBuf, xs, computeStrides(xs), 0));
                }
                if (gamma.requiresGrad || gamma.gradFn != null) {
                    float[] dg = new float[d];
                    for (int i = 0; i < b; i++) {
                        for (int j = 0; j < d; j++) {
                            dg[j] += go[i * d + j] * hNorm[i * d + j];
                        }
                    }
                    GpuBuffer dgBuf = device.allocate((long) dg.length * 4L);
                    dgBuf.copyFromHost(dg, 0, 0, dg.length);
                    gamma.propagateGrad(new GpuTensor(dgBuf, gamma.data.shape(),
                        computeStrides(gamma.data.shape()), 0));
                }
                if (beta.requiresGrad || beta.gradFn != null) {
                    float[] db = new float[d];
                    for (int i = 0; i < b; i++) {
                        for (int j = 0; j < d; j++) {
                            db[j] += go[i * d + j];
                        }
                    }
                    GpuBuffer dbBuf = device.allocate((long) db.length * 4L);
                    dbBuf.copyFromHost(db, 0, 0, db.length);
                    beta.propagateGrad(new GpuTensor(dbBuf, beta.data.shape(),
                        computeStrides(beta.data.shape()), 0));
                }
            };
            if (gamma != this) captureChildren(out, this, gamma);
            else captureChild(out, this);
        }
        return out;
    }

    public GpuAutogradTensor expand(int... targetShape) {
        GpuTensorMath math = new GpuTensorMath(device);
        GpuTensor r = math.expand(this.data, targetShape);
        GpuAutogradTensor out = new GpuAutogradTensor(r, device);
        if (this.requiresGrad || this.gradFn != null) {
            int[] origShape = this.data.shape();
            int aNd = origShape.length;
            int cNd = targetShape.length;
            int[] origStrides = this.data.strides();
            out.gradFn = (gradOutput) -> {
                float[] hg = gradOutput.toHost();
                int origNumel = numelFromShape(origShape);
                float[] hc = new float[origNumel];
                java.util.Arrays.fill(hc, 0f);
                for (int flat = 0; flat < hg.length; flat++) {
                    int idx = flat;
                    int aIdx = 0;
                    for (int d = cNd - 1; d >= 0; d--) {
                        int cSize = targetShape[d];
                        int pos = idx % cSize;
                        idx /= cSize;
                        if (d >= cNd - aNd) {
                            int aDim = d - (cNd - aNd);
                            if (origShape[aDim] > 1) {
                                aIdx += pos * origStrides[aDim];
                            }
                        }
                    }
                    hc[aIdx] += hg[flat];
                }
                GpuBuffer buf = device.allocate((long) hc.length * 4L);
                buf.copyFromHost(hc, 0, 0, hc.length);
                GpuTensor gg = new GpuTensor(buf, origShape,
                    computeStrides(origShape), 0);
                this.propagateGrad(gg);
            };
            captureChild(out, this);
        }
        return out;
    }

    private static int numelFromShape(int[] shape) {
        int n = 1;
        for (int d : shape) n *= d;
        return n;
    }

    private GpuTensor transposeCopy(GpuTensor t, int rows, int cols) {
        float[] h = t.toHost();
        float[] ht = new float[h.length];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                ht[c * rows + r] = h[r * cols + c];
            }
        }
        GpuBuffer newBuf = device.allocate((long) ht.length * 4L);
        newBuf.copyFromHost(ht, 0, 0, ht.length);
        return new GpuTensor(newBuf, new int[]{cols, rows},
            new int[]{1, cols}, 0);
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int s = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = s;
            s *= shape[i];
        }
        return strides;
    }

    private static void captureChild(GpuAutogradTensor out, GpuAutogradTensor child) {
        out.children.clear();
        out.children.add(child);
    }

    private static void captureChildren(GpuAutogradTensor out, GpuAutogradTensor a, GpuAutogradTensor b) {
        out.children.clear();
        out.children.add(a);
        out.children.add(b);
    }

    @FunctionalInterface
    private interface GradFn {
        void apply(GpuTensor gradOutput);
    }
}
