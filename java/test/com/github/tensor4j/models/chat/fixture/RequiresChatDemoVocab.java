/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.fixture;

import org.junit.jupiter.api.extension.ConditionEvaluationResult;
import org.junit.jupiter.api.extension.ExecutionCondition;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.Arrays;
import java.util.stream.Collectors;

/** Skip when llama3.2 vocab fixture JSON is missing (run {@code capture_tinygrad_chat_vocab.py}). */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(RequiresChatDemoVocab.RequiresChatDemoVocabCondition.class)
public @interface RequiresChatDemoVocab {

    ChatDemoVocabMode[] value();

    final class RequiresChatDemoVocabCondition implements ExecutionCondition {

        @Override
        public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
            RequiresChatDemoVocab annotation = context.getElement()
                    .map(element -> element.getAnnotation(RequiresChatDemoVocab.class))
                    .orElse(null);
            if (annotation == null) {
                annotation = context.getTestClass()
                        .map(type -> type.getAnnotation(RequiresChatDemoVocab.class))
                        .orElse(null);
            }
            if (annotation == null) {
                return ConditionEvaluationResult.enabled("no annotation");
            }
            for (ChatDemoVocabMode mode : annotation.value()) {
                if (!ChatDemoVocab.resourcePresent(mode)) {
                    return ConditionEvaluationResult.disabled(
                            mode + " vocab missing — run: python tools/capture_tinygrad_chat_vocab.py --mode "
                                    + (mode == ChatDemoVocabMode.FULL ? "full" : "pruned"));
                }
            }
            String modes = Arrays.stream(annotation.value()).map(Enum::name).collect(Collectors.joining(", "));
            return ConditionEvaluationResult.enabled(modes + " vocab fixture(s) present");
        }
    }
}
