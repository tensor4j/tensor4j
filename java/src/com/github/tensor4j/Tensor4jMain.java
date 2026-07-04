/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j;

import java.awt.GraphicsEnvironment;
import javax.swing.SwingUtilities;
import com.github.tensor4j.cli.CliRunner;
import com.github.tensor4j.models.algebra.AlgebraModel;
import com.github.tensor4j.ui.Tensor4jFrame;

/**
 * tensor4j sample entry — CLI when args are present, Swing GUI otherwise.
 */
public final class Tensor4jMain {

    public static final String SUCCESS_MARKER = "ITW_SAMPLE_TENSOR4J_SUCCESS";

    private Tensor4jMain() {
    }

    public static void main(String[] args) {
        AlgebraModel model = new AlgebraModel();
        if (args.length > 0 && !isGuiOnly(args)) {
            int code = new CliRunner(model, System.out, System.err).run(args);
            System.out.println(SUCCESS_MARKER + " cli exit=" + code);
            System.out.flush();
            System.exit(code);
        }

        if (GraphicsEnvironment.isHeadless()) {
            int code = new CliRunner(model, System.out, System.err).run(new String[] {"infer"});
            System.out.println(SUCCESS_MARKER + " headless jdk=" + System.getProperty("java.version"));
            System.out.flush();
            System.exit(code);
        }

        SwingUtilities.invokeLater(new GuiLauncher());
    }

    private static boolean isGuiOnly(String[] args) {
        return args.length == 1 && "gui".equalsIgnoreCase(args[0]);
    }
}
