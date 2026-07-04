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

import java.util.ArrayList;
import java.util.List;
import com.github.tensor4j.core.Tensor;

/** Sequential stack of modules (tinygrad {@code nn.Sequential}). */
public final class Sequential extends Module {

    private final List<Layer> layers = new ArrayList<>();

    public Sequential add(String name, Module module) {
        layers.add(new Layer(name, module));
        return this;
    }

    public List<Layer> layers() {
        return List.copyOf(layers);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (Layer layer : layers) {
            current = layer.module().forward(current);
        }
        return current;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.module().parameters());
        }
        return params;
    }

    public Linear findLinear(String name) {
        for (Layer layer : layers) {
            if (layer.name().equals(name) && layer.module() instanceof Linear linear) {
                return linear;
            }
        }
        throw new IllegalArgumentException("linear layer not found: " + name);
    }

    public record Layer(String name, Module module) {
    }
}
