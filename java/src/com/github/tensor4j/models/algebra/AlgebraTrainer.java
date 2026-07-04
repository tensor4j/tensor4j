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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import com.github.tensor4j.autograd.GradFlow;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.optim.Sgd;

/** Java-side training loop (tinygrad exports weights; Java can fine-tune). */
public final class AlgebraTrainer {

    private final AlgebraModel model;
    private final Random random = new Random(42);

    public AlgebraTrainer(AlgebraModel model) {
        this.model = model;
    }

    public List<Float> train(int epochs, float learningRate, int batchSize) {
        Sgd optimizer = new Sgd(learningRate, 1f);
        GradFlow flow = new GradFlow();
        List<Float> losses = new ArrayList<>();
        for (int epoch = 0; epoch < epochs; epoch++) {
            float batchLoss = 0f;
            for (int sample = 0; sample < batchSize; sample++) {
                AlgebraEquation eq = randomEquation();
                Tensor input = Tensor.of(eq.normalizedFeatures(), 1, 3);
                Tensor target = Tensor.of(new float[] {eq.exactSolution()}, 1, 1);
                model.network().zeroGrad();
                Tensor prediction = model.network().forward(input);
                Tensor loss = GradFlow.mseLoss(prediction, target, flow);
                loss.backward();
                optimizer.step(model.network().parameters());
                batchLoss += loss.data()[0];
            }
            losses.add(batchLoss / batchSize);
        }
        return losses;
    }

    private AlgebraEquation randomEquation() {
        float a = random.nextInt(9) + 1;
        if (random.nextBoolean()) {
            a = -a;
        }
        float b = random.nextInt(21) - 10;
        float x = random.nextInt(21) - 10;
        float c = a * x + b;
        return new AlgebraEquation(a, b, c, a + "x+" + b + "=" + c);
    }
}
