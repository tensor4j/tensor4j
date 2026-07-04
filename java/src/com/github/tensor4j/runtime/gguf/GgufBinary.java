/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.gguf;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** Little-endian GGUF primitive read/write (gguf.cpp gguf_reader). */
final class GgufBinary {

    private GgufBinary() {
    }

    static ByteBuffer wrapRead(byte[] data) {
        return ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    }

    static ByteBuffer allocateWrite(int capacity) {
        return ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
    }

    static void expectMagic(ByteBuffer buffer) {
        for (byte expected : GgufConstants.MAGIC) {
            if (buffer.get() != expected) {
                throw new IllegalArgumentException("invalid GGUF magic");
            }
        }
    }

    static void writeMagic(ByteBuffer buffer) {
        buffer.put(GgufConstants.MAGIC);
    }

    static String readString(ByteBuffer buffer) {
        long length = buffer.getLong();
        if (length < 0 || length > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("invalid string length " + length);
        }
        byte[] bytes = new byte[(int) length];
        buffer.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    static void writeString(ByteBuffer buffer, String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        buffer.putLong(bytes.length);
        buffer.put(bytes);
    }

    static Object readScalar(ByteBuffer buffer, GgufType type) {
        return switch (type) {
            case UINT8 -> buffer.get() & 0xFF;
            case INT8 -> buffer.get();
            case UINT16 -> buffer.getShort() & 0xFFFF;
            case INT16 -> buffer.getShort();
            case UINT32 -> buffer.getInt() & 0xFFFF_FFFFL;
            case INT32 -> buffer.getInt();
            case FLOAT32 -> buffer.getFloat();
            case BOOL -> buffer.get() != 0;
            case STRING -> readString(buffer);
            case UINT64 -> buffer.getLong();
            case INT64 -> buffer.getLong();
            case FLOAT64 -> buffer.getDouble();
            case ARRAY -> throw new IllegalArgumentException("ARRAY is not a scalar");
        };
    }

    static void writeScalar(ByteBuffer buffer, GgufType type, Object value) {
        switch (type) {
            case UINT8 -> buffer.put(((Number) value).byteValue());
            case INT8 -> buffer.put(((Number) value).byteValue());
            case UINT16 -> buffer.putShort(((Number) value).shortValue());
            case INT16 -> buffer.putShort(((Number) value).shortValue());
            case UINT32 -> buffer.putInt(((Number) value).intValue());
            case INT32 -> buffer.putInt(((Number) value).intValue());
            case FLOAT32 -> buffer.putFloat(((Number) value).floatValue());
            case BOOL -> buffer.put((byte) ((Boolean) value ? 1 : 0));
            case STRING -> writeString(buffer, (String) value);
            case UINT64, INT64 -> buffer.putLong(((Number) value).longValue());
            case FLOAT64 -> buffer.putDouble(((Number) value).doubleValue());
            case ARRAY -> throw new IllegalArgumentException("ARRAY is not a scalar");
        }
    }

    static Object readValue(ByteBuffer buffer) {
        GgufType type = GgufType.fromId(buffer.getInt());
        return readValueBody(buffer, type);
    }

    static Object readValueBody(ByteBuffer buffer, GgufType type) {
        if (type == GgufType.ARRAY) {
            GgufType elementType = GgufType.fromId(buffer.getInt());
            long count = buffer.getLong();
            if (count < 0 || count > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("invalid array length " + count);
            }
            Object[] elements = new Object[(int) count];
            for (int i = 0; i < count; i++) {
                elements[i] = readScalar(buffer, elementType);
            }
            return new GgufArrayValue(elementType, elements);
        }
        return readScalar(buffer, type);
    }

    static void writeValue(ByteBuffer buffer, GgufType type, Object value) {
        buffer.putInt(type.id());
        if (type == GgufType.ARRAY) {
            GgufArrayValue array = (GgufArrayValue) value;
            buffer.putInt(array.elementType().id());
            buffer.putLong(array.elements().length);
            for (Object element : array.elements()) {
                writeScalar(buffer, array.elementType(), element);
            }
            return;
        }
        writeScalar(buffer, type, value);
    }
}
