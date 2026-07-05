/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat;

/** Why {@link ChatGenerator} ended a decode loop. */
public enum ChatGenerationStopReason {

    EMPTY_PROMPT,
    EOS,
    EOT,
    MAX_TOKENS,
    UNKNOWN;

    static ChatGenerationStopReason forEndToken(ChatTokenizer tokenizer, int tokenId) {
        // Llama 3 GGUF often sets eos_id == eot_id (128009); chat turns stop on <|eot_id|>.
        if (tokenizer.eotId() >= 0 && tokenId == tokenizer.eotId()) {
            return EOT;
        }
        if (tokenId == tokenizer.eosId()) {
            return EOS;
        }
        if (tokenizer.isEndOfGeneration(tokenId)) {
            return EOT;
        }
        return UNKNOWN;
    }
}
