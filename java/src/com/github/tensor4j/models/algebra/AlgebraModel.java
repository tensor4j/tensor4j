/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.algebra;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.io.ModelLoader;
import com.github.tensor4j.io.WeightFormat;
import com.github.tensor4j.nn.Linear;
import com.github.tensor4j.nn.Relu;
import com.github.tensor4j.nn.Sequential;

/** MLP inference head for {@code ax + b = c} (primary inference target). */
public final class AlgebraModel {

    public static final String MODEL_RESOURCE_JSON = "/models/algebra-v1.t4j.json";
    public static final String MODEL_RESOURCE_SAFETENSORS = "/models/algebra-v1.safetensors";

    private final Sequential network;
    private WeightFormat weightFormat = WeightFormat.fromSystemProperty();

    public AlgebraModel() {
        network = new Sequential()
                .add("fc1", new Linear(3, 16, "fc1"))
                .add("relu1", new Relu())
                .add("fc2", new Linear(16, 16, "fc2"))
                .add("relu2", new Relu())
                .add("fc3", new Linear(16, 1, "fc3"));
    }

    public Sequential network() {
        return network;
    }

    public WeightFormat weightFormat() {
        return weightFormat;
    }

    public void setWeightFormat(WeightFormat weightFormat) {
        this.weightFormat = weightFormat;
    }

    public void loadBundledWeights() throws IOException {
        String resource = weightFormat == WeightFormat.SAFETENSORS
                ? MODEL_RESOURCE_SAFETENSORS
                : MODEL_RESOURCE_JSON;
        Map<String, Tensor> tensors = ModelLoader.loadResource(resource, weightFormat);
        ModelLoader.applyToSequential(network, tensors);
    }

    public void loadWeights(Path path) throws IOException {
        Map<String, Tensor> tensors = ModelLoader.load(path, WeightFormat.fromPath(path));
        ModelLoader.applyToSequential(network, tensors);
    }

    public float predict(AlgebraEquation equation) {
        Tensor input = Tensor.of(equation.normalizedFeatures(), 1, 3);
        Tensor output = network.forward(input);
        return output.data()[0];
    }

    public AlgebraResult infer(String equationText) {
        AlgebraEquation equation = AlgebraEquation.parse(equationText);
        float predicted = predict(equation);
        float exact = equation.exactSolution();
        return new AlgebraResult(equation, predicted, exact);
    }

    public record AlgebraResult(AlgebraEquation equation, float predicted, float exact) {

        public float error() {
            return Math.abs(predicted - exact);
        }
    }
}
