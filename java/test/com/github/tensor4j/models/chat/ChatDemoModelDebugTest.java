package com.github.tensor4j.models.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import java.util.Random;
import org.junit.jupiter.api.Test;

class ChatDemoModelDebugTest {

    @Test
    void helloCompletesWithThereUnderQualitySampling() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatTokenizer tok = model.tokenizer();
        ChatGenerationOptions options = ChatGenerationOptions.quality(tok);
        model.resetCache();
        float[] logits = model.forward(tok.encode("Hello"));
        assertEquals(2, ChatSampler.argmax(logits));

        Random rng = new Random(42);
        StringBuilder completion = new StringBuilder();
        int generated = 0;
        for (int step = 0; step < options.maxNewTokens(); step++) {
            int next = ChatSampler.sample(logits, options, generated, rng);
            if (next == tok.eosId() && generated >= options.minNewTokens()) {
                break;
            }
            if (next == tok.bosId() || next == tok.eosId()) {
                logits = model.forward(new int[] {next});
                continue;
            }
            completion.append(tok.decode(new int[] {next}));
            generated++;
            logits = model.forward(new int[] {next});
        }
        assertTrue(completion.toString().contains("there"), completion::toString);
    }
}
