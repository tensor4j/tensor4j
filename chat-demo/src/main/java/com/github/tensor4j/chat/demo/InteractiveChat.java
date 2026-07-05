/*

 * Copyright 2026 Tensor4j Maintainers

 *

 * Licensed under the Apache License, Version 2.0 (the "License");

 * you may not use this file except in compliance with the License.

 * You may obtain a copy of the License at

 *

 *     http://www.apache.org/licenses/LICENSE-2.0

 */

package com.github.tensor4j.chat.demo;



import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatGenerationResult;
import com.github.tensor4j.models.chat.ChatGenerator;
import com.github.tensor4j.models.chat.ChatHistoryMode;
import com.github.tensor4j.models.chat.InferCompatMode;

import com.github.tensor4j.models.chat.ChatModel;

import com.github.tensor4j.models.chat.ChatTemplate;

import com.github.tensor4j.runtime.gguf.MmappedGgufFile;

import java.nio.charset.StandardCharsets;

import java.nio.file.Files;

import java.nio.file.Path;

import java.nio.file.Paths;

import java.time.LocalDateTime;

import java.time.format.DateTimeFormatter;

import java.util.Locale;

import java.util.Scanner;



/** Interactive REPL against a mmap'd GGUF; transcripts written incrementally. */

public final class InteractiveChat {



    private static final DateTimeFormatter FILE_STAMP =

            DateTimeFormatter.ofPattern("yyyy-MM-dd_HHmmss", Locale.ROOT);



    public static void main(String[] args) throws Exception {

        String ggufPath = requireEnv("TENSOR4J_GGUF_PATH");

        Path saveDir = resolveSaveDir();

        Files.createDirectories(saveDir);

        String fileBase = FILE_STAMP.format(LocalDateTime.now()) + "-" + resolveLogBase(ggufPath);

        try (MmappedGgufFile mapped = MmappedGgufFile.open(Paths.get(ggufPath));

                Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8)) {

            ChatModel model = ChatModel.fromGguf(mapped);
            ChatTemplate template = ChatTemplate.fromEnvironmentOrTokenizer(model.tokenizer());
            ChatHistoryMode historyMode = ChatHistoryMode.fromEnvironment();
            printBanner(ggufPath, template, historyMode, saveDir, fileBase);

            ChatGenerationOptions options = ChatGenerationOptions.fromEnvironment(model.tokenizer());

            ChatGenerator generator = new ChatGenerator(model, options, historyMode);



            try (ChatSessionLogger logger = new ChatSessionLogger(

                    saveDir, fileBase, ggufPath, template, options, model.tokenizer())) {

                System.out.println("Logs (updated each turn):");

                System.out.println("  " + logger.markdownPath().toAbsolutePath());

                System.out.println("  " + logger.auditPath().toAbsolutePath());

                System.out.println();

                System.out.println("Type a message and press Enter. Commands: /help /reset /quit");

                System.out.println();



                while (true) {

                    System.out.print("you> ");

                    System.out.flush();

                    if (!scanner.hasNextLine()) {

                        break;

                    }

                    String line = scanner.nextLine();

                    if (line == null) {

                        break;

                    }

                    String trimmed = line.trim();

                    if (trimmed.isEmpty()) {

                        continue;

                    }

                    if (handleCommand(trimmed, generator, logger)) {

                        break;

                    }



                    var tokenizer = model.tokenizer();

                    int[] userTurnIds = template.encodeUserTurn(tokenizer, line);

                    int[] assistantPrimeIds = template.encodeAssistantPrime(tokenizer);

                    int[] promptIds = generator.planPromptIds(line, template);

                    int[] sessionBefore = generator.sessionTokenIds();

                    int[] cachedBefore = generator.cachedTokenIds();

                    int kvBefore = generator.kvLength();



                    System.out.print("assistant> ");
                    System.out.flush();
                    ChatGenerationResult result = generator.continueConversationStreaming(
                            line, template, piece -> {
                                System.out.print(piece);
                                System.out.flush();
                            });
                    System.out.println();

                    System.out.printf(
                            Locale.US,
                            "  [%d new tokens, stop=%s%s, %d left, %s, session %d tokens, kv %d]%n%n",
                            result.tokenCount(),
                            result.stopReason().name().toLowerCase(Locale.ROOT),
                            result.stopTokenId() >= 0 ? " id=" + result.stopTokenId() : "",
                            result.tokensRemaining(),
                            historyMode == ChatHistoryMode.LLAMA && result.prefixReuseTokens() == 0
                                    ? "delta-only"
                                    : result.prefixReuseTokens() + " prefix reused",
                            generator.sessionTokenIds().length,
                            generator.kvLength());



                    logger.logTurn(

                            logger.nextTurnNumber(),

                            line,

                            userTurnIds,

                            assistantPrimeIds,

                            promptIds,

                            sessionBefore,

                            cachedBefore,

                            kvBefore,

                            result,

                            generator);

                }

            }

        }

    }



    private static boolean handleCommand(String line, ChatGenerator generator, ChatSessionLogger logger) {

        if (!line.startsWith("/")) {

            return false;

        }

        String cmd = line.toLowerCase(Locale.ROOT);

        switch (cmd) {

            case "/quit", "/exit", "/q" -> {

                return true;

            }

            case "/reset" -> {

                generator.resetSession();

                logger.logSystem("(session reset — KV cache and conversation history cleared)");

                System.out.println("(session reset)");

                return false;

            }

            case "/help" -> {

                printHelp();

                return false;

            }

            default -> {

                System.out.println("Unknown command. Try /help /reset /quit");

                return false;

            }

        }

    }



    private static void printHelp() {

        System.out.println("""

                Commands:

                  /help   — this message

                  /reset  — clear KV cache and conversation history

                  /quit   — exit (logs already saved each turn)



                Environment (set before launch or via chat.sh):

                  TENSOR4J_GGUF_PATH      path to .gguf (chat.sh defaults to Qwen2.5-1.5B-Instruct)

                  TENSOR4J_CHAT_TEMPLATE  qwen2 (default via chat.sh), llama3, or plain; auto from GGUF if unset

                  TENSOR4J_CHAT_LOG_BASE  transcript suffix (default qwen25-1.5b; inferred from GGUF name)

                  TENSOR4J_CHAT_MODE      quality (default) or greedy

                  TENSOR4J_CHAT_MAX_TOKENS  max new tokens per reply (default 256)

                  TENSOR4J_CHAT_TEMPERATURE TENSOR4J_CHAT_TOP_P TENSOR4J_CHAT_TOP_K

                  TENSOR4J_CHAT_SAVE_DIR  transcript directory (default ~/.local/conversations)

                  TENSOR4J_CHAT_DEBUG          true = per-token stderr trace + stop summary

                  TENSOR4J_INFER_COMPAT        tinygrad (parity) or llama (default; llama.cpp chat fixes)
                  TENSOR4J_CHAT_HISTORY_MODE  delta (llama.cpp simple-chat, default) or legacy (full replay)

                  TENSOR4J_CHAT_NO_KV_REUSE  true = full prefill each turn (legacy debug)

                """);

    }



    private static void printBanner(
            String ggufPath, ChatTemplate template, ChatHistoryMode historyMode, Path saveDir, String fileBase) {

        System.out.println("tensor4j interactive chat");

        System.out.println("  model    : " + ggufPath);

        System.out.println("  template : " + template.name().toLowerCase(Locale.ROOT));

        System.out.println("  history  : " + formatHistoryMode(historyMode));

        System.out.println("  compat   : " + formatInferCompat(InferCompatMode.fromEnvironment()));

        System.out.println("  save dir : " + saveDir.toAbsolutePath());

        System.out.println("  log base : " + fileBase);

        System.out.println();

    }



    private static String requireEnv(String key) {

        String value = System.getenv(key);

        if (value == null || value.isBlank()) {

            System.err.println(
                    "Missing "
                            + key
                            + " — set path to a GGUF model (e.g. run chat.sh --download for Qwen2.5-1.5B).");

            System.exit(1);

        }

        return value.trim();

    }



    private static Path resolveSaveDir() {

        String raw = System.getenv("TENSOR4J_CHAT_SAVE_DIR");

        if (raw != null && !raw.isBlank()) {

            return Paths.get(raw.trim());

        }

        String home = System.getenv("HOME");

        if (home == null || home.isBlank()) {

            home = System.getProperty("user.home");

        }

        return Paths.get(home, ".local", "conversations");

    }

    private static String formatInferCompat(InferCompatMode mode) {
        if (mode == InferCompatMode.TINYGRAD) {
            return "tinygrad (parity reference; no llama.cpp chat fixes)";
        }
        return "llama.cpp (delta KV, sampled assistant ids, eot guards)";
    }

    private static String formatHistoryMode(ChatHistoryMode historyMode) {
        if (historyMode == ChatHistoryMode.LLAMA) {
            return "delta (llama.cpp simple-chat; prompt format follows template above)";
        }
        return "legacy (tinygrad full token replay)";
    }

    private static String resolveLogBase(String ggufPath) {
        String env = System.getenv("TENSOR4J_CHAT_LOG_BASE");
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        String name = Paths.get(ggufPath).getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.contains("llama")) {
            return "llama32";
        }
        if (name.contains("qwen")) {
            return "qwen25-1.5b";
        }
        return "qwen25-1.5b";
    }

}


