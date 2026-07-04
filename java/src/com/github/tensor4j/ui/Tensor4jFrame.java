/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.KeyEvent;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Locale;
import javax.swing.BorderFactory;
import javax.swing.InputMap;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.KeyStroke;
import javax.swing.UIManager;
import com.github.tensor4j.cli.CliRunner;
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.manifold.EuclideanManifold;
import com.github.tensor4j.manifold.ManifoldPoint;
import com.github.tensor4j.models.algebra.AlgebraModel;
import com.github.tensor4j.models.algebra.AlgebraModel.AlgebraResult;
import com.github.tensor4j.models.algebra.AlgebraTrainer;

public final class Tensor4jFrame extends JFrame {

    private final AlgebraModel model = new AlgebraModel();
    private final JTextArea log = new JTextArea();
    private final JTextField equationField = new JTextField("2x + 3 = 11", 28);
    private final JLabel inferResult = new JLabel(" ");
    private final ManifoldCanvas manifoldCanvas = new ManifoldCanvas();

    public Tensor4jFrame() {
        super("tensor4j — Tensor4j sample");
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {
            // keep default
        }
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setMinimumSize(new Dimension(920, 640));
        buildUi();
        bootstrapModel();
    }

    private void buildUi() {
        JPanel root = new JPanel(new BorderLayout());
        root.setBackground(new Color(248, 250, 252));
        root.add(buildHeader(), BorderLayout.NORTH);

        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("Inference", buildInferencePanel());
        tabs.addTab("Training", buildTrainingPanel());
        tabs.addTab("Tensor", buildTensorPanel());
        tabs.addTab("Manifold", buildManifoldPanel());
        tabs.addTab("CLI", buildCliPanel());
        root.add(tabs, BorderLayout.CENTER);

        log.setEditable(false);
        log.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane logScroll = new JScrollPane(log);
        logScroll.setPreferredSize(new Dimension(100, 120));
        logScroll.setBorder(BorderFactory.createTitledBorder("Session log"));
        root.add(logScroll, BorderLayout.SOUTH);
        setContentPane(root);
    }

    private JPanel buildHeader() {
        JPanel header = new JPanel(new BorderLayout());
        header.setBackground(new Color(15, 23, 42));
        header.setBorder(BorderFactory.createEmptyBorder(18, 22, 18, 22));
        JLabel title = new JLabel("tensor4j");
        title.setForeground(Color.WHITE);
        title.setFont(title.getFont().deriveFont(Font.BOLD, 24f));
        JLabel subtitle = new JLabel("Tensor · autograd · gradient flow · manifold · inference");
        subtitle.setForeground(new Color(148, 163, 184));
        subtitle.setFont(subtitle.getFont().deriveFont(Font.PLAIN, 13f));
        JPanel text = new JPanel(new BorderLayout());
        text.setOpaque(false);
        text.add(title, BorderLayout.NORTH);
        text.add(subtitle, BorderLayout.SOUTH);
        header.add(text, BorderLayout.WEST);
        return header;
    }

    private JPanel buildInferencePanel() {
        JPanel panel = paddedPanel();
        panel.add(label("High-school algebra model (ax + b = c)"), BorderLayout.NORTH);
        JPanel row = new JPanel(new BorderLayout(8, 0));
        row.setOpaque(false);
        row.add(equationField, BorderLayout.CENTER);
        JButton run = primaryButton("Run inference");
        run.addActionListener(new InferenceAction());
        row.add(run, BorderLayout.EAST);
        panel.add(row, BorderLayout.CENTER);
        inferResult.setBorder(BorderFactory.createEmptyBorder(12, 0, 0, 0));
        panel.add(inferResult, BorderLayout.SOUTH);
        return panel;
    }

    private JPanel buildTrainingPanel() {
        JPanel panel = paddedPanel();
        JTextField epochs = new JTextField("60", 6);
        JTextField lr = new JTextField("0.05", 6);
        JButton train = primaryButton("Train in Java");
        train.addActionListener(new TrainingAction(epochs, lr));
        JPanel controls = new JPanel();
        controls.setOpaque(false);
        controls.add(new JLabel("epochs"));
        controls.add(epochs);
        controls.add(new JLabel("lr"));
        controls.add(lr);
        controls.add(train);
        panel.add(controls, BorderLayout.NORTH);
        panel.add(new JLabel("<html><p style='width:520px;color:#475569'>Fine-tune on-device with autograd + SGD. "
                + "Export weights from tinygrad using <code>tools/export_algebra_model.py</code> "
                + "for production inference bundles.</p></html>"), BorderLayout.CENTER);
        return panel;
    }

    private JPanel buildTensorPanel() {
        JPanel panel = paddedPanel();
        JTextArea area = new JTextArea(6, 40);
        area.setText("shape [2, 2]\nrandom normal tensor");
        JButton create = primaryButton("Materialize tensor");
        create.addActionListener(new TensorCreateAction(area));
        panel.add(new JScrollPane(area), BorderLayout.CENTER);
        panel.add(create, BorderLayout.SOUTH);
        return panel;
    }

    private JPanel buildManifoldPanel() {
        JPanel panel = paddedPanel();
        manifoldCanvas.setPreferredSize(new Dimension(520, 280));
        panel.add(manifoldCanvas, BorderLayout.CENTER);
        JButton flow = primaryButton("Trace gradient flow");
        flow.addActionListener(new ManifoldFlowAction());
        panel.add(flow, BorderLayout.SOUTH);
        return panel;
    }

    private JPanel buildCliPanel() {
        JPanel panel = paddedPanel();
        JTextField command = new JTextField("infer --equation \"2x + 3 = 11\"", 40);
        JTextArea output = new JTextArea(8, 40);
        output.setEditable(false);
        output.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JButton run = primaryButton("Execute");
        run.addActionListener(new CliExecuteAction(command, output, run));
        InputMap input = command.getInputMap(JComponent.WHEN_FOCUSED);
        input.put(KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, 0), "run");
        command.getActionMap().put("run", new EnterRunAction(run));
        panel.add(command, BorderLayout.NORTH);
        panel.add(new JScrollPane(output), BorderLayout.CENTER);
        panel.add(run, BorderLayout.SOUTH);
        return panel;
    }

    private void runInference() {
        try {
            AlgebraResult result = model.infer(equationField.getText().trim());
            inferResult.setText(String.format(Locale.US,
                    "<html><b>Predicted x:</b> %.4f &nbsp;·&nbsp; <b>Exact:</b> %.4f &nbsp;·&nbsp; <b>|error|:</b> %.6f</html>",
                    result.predicted(), result.exact(), result.error()));
            appendLog(String.format(Locale.US, "infer %s => pred=%.4f exact=%.4f%n",
                    equationField.getText(), result.predicted(), result.exact()));
        } catch (Exception ex) {
            inferResult.setText("Error: " + ex.getMessage());
        }
    }

    private void bootstrapModel() {
        try {
            model.loadBundledWeights();
            appendLog("loaded bundled algebra-v1 weights\n");
        } catch (IOException ex) {
            appendLog("no bundled weights; run Training tab or CLI train first\n");
        }
    }

    private void appendLog(String text) {
        log.append(text);
        log.setCaretPosition(log.getDocument().getLength());
    }

    private static JPanel paddedPanel() {
        JPanel panel = new JPanel(new BorderLayout(8, 8));
        panel.setOpaque(false);
        panel.setBorder(BorderFactory.createEmptyBorder(16, 16, 16, 16));
        return panel;
    }

    private static JLabel label(String text) {
        JLabel label = new JLabel(text);
        label.setFont(label.getFont().deriveFont(Font.BOLD, 14f));
        return label;
    }

    private static JButton primaryButton(String text) {
        JButton button = new JButton(text);
        button.setBackground(new Color(37, 99, 235));
        button.setForeground(Color.WHITE);
        button.setFocusPainted(false);
        button.setBorder(BorderFactory.createEmptyBorder(8, 14, 8, 14));
        return button;
    }

    private final class InferenceAction implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent event) {
            runInference();
        }
    }

    private final class TrainingAction implements ActionListener {
        private final JTextField epochsField;
        private final JTextField lrField;

        TrainingAction(JTextField epochsField, JTextField lrField) {
            this.epochsField = epochsField;
            this.lrField = lrField;
        }

        @Override
        public void actionPerformed(ActionEvent event) {
            int n = Integer.parseInt(epochsField.getText().trim());
            float rate = Float.parseFloat(lrField.getText().trim());
            AlgebraTrainer trainer = new AlgebraTrainer(model);
            List<Float> losses = trainer.train(n, rate, 16);
            appendLog(String.format(Locale.US, "training done: loss %.4f to %.4f%n",
                    losses.get(0), losses.get(losses.size() - 1)));
        }
    }

    private final class TensorCreateAction implements ActionListener {
        private final JTextArea area;

        TensorCreateAction(JTextArea area) {
            this.area = area;
        }

        @Override
        public void actionPerformed(ActionEvent event) {
            Tensor t = Tensor.randn(2, 2);
            area.setText(t.toString());
            appendLog("tensor created " + t + "\n");
        }
    }

    private final class ManifoldFlowAction implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent event) {
            EuclideanManifold manifold = new EuclideanManifold();
            ManifoldPoint start = manifold.point(Tensor.of(new float[] {2f, -1f}, 2));
            ManifoldPoint end = manifold.gradientFlow(
                    start, Tensor.of(new float[] {1f, 2f}, 2), 8, 0.15f).get(7);
            manifoldCanvas.setPoints(start.coordinates(), end.coordinates());
            appendLog("manifold gradient flow traced (8 steps)\n");
        }
    }

    private final class CliExecuteAction implements ActionListener {
        private final JTextField commandField;
        private final JTextArea outputArea;

        CliExecuteAction(JTextField commandField, JTextArea outputArea, JButton ignoredTrigger) {
            this.commandField = commandField;
            this.outputArea = outputArea;
        }

        @Override
        public void actionPerformed(ActionEvent event) {
            String[] tokens = ("tensor4j " + commandField.getText()).split("\\s+");
            String[] args = new String[Math.max(0, tokens.length - 1)];
            System.arraycopy(tokens, 1, args, 0, args.length);
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            CliRunner runner = new CliRunner(model, new PrintStream(buf), System.err);
            runner.run(args);
            outputArea.setText(buf.toString());
        }
    }

    private static final class EnterRunAction extends javax.swing.AbstractAction {
        private final JButton trigger;

        EnterRunAction(JButton trigger) {
            this.trigger = trigger;
        }

        @Override
        public void actionPerformed(ActionEvent event) {
            trigger.doClick();
        }
    }

    private static final class ManifoldCanvas extends JPanel {

        private float[] start = {0.2f, 0.5f};
        private float[] end = {0.8f, 0.7f};

        void setPoints(Tensor startTensor, Tensor endTensor) {
            start = new float[] {startTensor.data()[0], startTensor.data()[1]};
            end = new float[] {endTensor.data()[0], endTensor.data()[1]};
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            int w = getWidth();
            int h = getHeight();
            g2.setColor(new Color(226, 232, 240));
            g2.fillRect(0, 0, w, h);
            int x0 = (int) (start[0] * w);
            int y0 = (int) (start[1] * h);
            int x1 = (int) (end[0] * w);
            int y1 = (int) (end[1] * h);
            g2.setColor(new Color(59, 130, 246));
            g2.drawLine(x0, y0, x1, y1);
            g2.fillOval(x0 - 6, y0 - 6, 12, 12);
            g2.setColor(new Color(16, 185, 129));
            g2.fillOval(x1 - 6, y1 - 6, 12, 12);
            g2.dispose();
        }
    }
}
