/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.ggml;

/** IEEE 754 half-precision helpers (ggml_fp16_t). */
public final class GgmlFp16 {

    private GgmlFp16() {
    }

    /** {@code GGML_FP16_TO_FP32} — little-endian uint16 at {@code offset}. */
    public static float toFloat32(byte[] data, int offset) {
        int bits = (data[offset] & 0xFF) | ((data[offset + 1] & 0xFF) << 8);
        return toFloat32(bits);
    }

    public static float toFloat32(int fp16Bits) {
        int sign = (fp16Bits & 0x8000) << 16;
        int exp = (fp16Bits >> 10) & 0x1F;
        int mant = fp16Bits & 0x3FF;
        if (exp == 0) {
            if (mant == 0) {
                return Float.intBitsToFloat(sign);
            }
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= 0x3FF;
        } else if (exp == 31) {
            if (mant == 0) {
                return Float.intBitsToFloat(sign | 0x7F800000);
            }
            return Float.intBitsToFloat(sign | 0x7F800000 | (mant << 13));
        }
        int fexp = exp + (127 - 15);
        return Float.intBitsToFloat(sign | (fexp << 23) | (mant << 13));
    }

    /** {@code GGML_FP32_TO_FP16} for reference quant tests. */
    public static short fromFloat32(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exp = (bits >>> 23) & 0xFF;
        int mant = bits & 0x7FFFFF;
        if (exp == 255) {
            if (mant != 0) {
                return (short) (sign | 0x7C00 | (mant >>> 13));
            }
            return (short) (sign | 0x7C00);
        }
        int newExp = exp - 127 + 15;
        if (newExp >= 31) {
            return (short) (sign | 0x7C00);
        }
        if (newExp <= 0) {
            if (newExp < -10) {
                return (short) sign;
            }
            mant |= 0x800000;
            int shift = 1 - newExp;
            mant = mant >>> shift;
            return (short) (sign | (mant >>> 13));
        }
        return (short) (sign | (newExp << 10) | (mant >>> 13));
    }

    public static void writeLittleEndian(byte[] data, int offset, short fp16) {
        data[offset] = (byte) (fp16 & 0xFF);
        data[offset + 1] = (byte) ((fp16 >>> 8) & 0xFF);
    }
}
