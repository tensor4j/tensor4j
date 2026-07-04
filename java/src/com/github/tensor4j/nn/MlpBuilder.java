/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.nn;

/** Build {@link Sequential} stacks from layer-size specs (tinygrad-style MLP). */
public final class MlpBuilder {

    private MlpBuilder() {
    }

    /** Fully-connected stack: {@code in -> h1 -> ... -> out} with ReLU between hidden layers. */
    public static Sequential fullyConnected(int[] layerSizes) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("need at least input and output sizes, e.g. 3,16,1");
        }
        Sequential network = new Sequential();
        for (int i = 0; i < layerSizes.length - 1; i++) {
            String name = "fc" + (i + 1);
            network.add(name, new Linear(layerSizes[i], layerSizes[i + 1], name));
            if (i < layerSizes.length - 2) {
                network.add("relu" + (i + 1), new Relu());
            }
        }
        return network;
    }

    public static int[] parseLayerSizes(String csv) {
        String[] parts = csv.split(",");
        int[] sizes = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            sizes[i] = Integer.parseInt(parts[i].trim());
        }
        return sizes;
    }
}
