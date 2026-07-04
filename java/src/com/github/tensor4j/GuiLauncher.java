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

import javax.swing.SwingUtilities;
import com.github.tensor4j.ui.Tensor4jFrame;

/** Starts the Swing UI on the EDT without lambdas. */
final class GuiLauncher implements Runnable {

    @Override
    public void run() {
        Tensor4jFrame frame = new Tensor4jFrame();
        frame.setVisible(true);
        System.out.println(Tensor4jMain.SUCCESS_MARKER + " gui jdk="
                + System.getProperty("java.version")
                + " vendor=" + System.getProperty("java.vendor"));
        System.out.flush();
    }
}
