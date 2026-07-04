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



        ChatTemplate template = ChatTemplate.fromEnvironment();

        String fileBase = FILE_STAMP.format(LocalDateTime.now()) + "-llama32";

        printBanner(ggufPath, template, saveDir, fileBase);



        try (MmappedGgufFile mapped = MmappedGgufFile.open(Paths.get(ggufPath));

                Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8)) {

            ChatModel model = ChatModel.fromGguf(mapped);

            ChatGenerationOptions options = ChatGenerationOptions.fromEnvironment(model.tokenizer());

            ChatGenerator generator = new ChatGenerator(model, options);



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

                            "  [%d new tokens, %d prefix reused, session %d tokens]%n%n",

                            result.tokenCount(),

                            result.prefixReuseTokens(),

                            generator.sessionTokenIds().length);



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

                  TENSOR4J_GGUF_PATH      path to .gguf model (required)

                  TENSOR4J_CHAT_TEMPLATE  llama3 (default) or plain

                  TENSOR4J_CHAT_MODE      quality (default) or greedy

                  TENSOR4J_CHAT_MAX_TOKENS  max new tokens per reply (default 256)

                  TENSOR4J_CHAT_TEMPERATURE TENSOR4J_CHAT_TOP_P TENSOR4J_CHAT_TOP_K

                  TENSOR4J_CHAT_SAVE_DIR  transcript directory (default ~/.local/conversations)

                  TENSOR4J_CHAT_NO_KV_REUSE  true = full prefill each turn (debug bleed)

                """);

    }



    private static void printBanner(String ggufPath, ChatTemplate template, Path saveDir, String fileBase) {

        System.out.println("tensor4j interactive chat");

        System.out.println("  model    : " + ggufPath);

        System.out.println("  template : " + template.name().toLowerCase(Locale.ROOT));

        System.out.println("  save dir : " + saveDir.toAbsolutePath());

        System.out.println("  log base : " + fileBase);

        System.out.println();

    }



    private static String requireEnv(String key) {

        String value = System.getenv(key);

        if (value == null || value.isBlank()) {

            System.err.println("Missing " + key + " — set path to Llama 3.2 GGUF file.");

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

}


