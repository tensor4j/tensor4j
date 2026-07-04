/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.algebra.demo;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import com.github.tensor4j.models.algebra.AlgebraEquation;

/** Step-by-step symbolic solution for {@code ax + b = c} (high-school algebra). */
final class WorkedSolution {

    private WorkedSolution() {
    }

    static List<String> steps(AlgebraEquation equation) {
        float a = equation.a();
        float b = equation.b();
        float c = equation.c();
        float x = equation.exactSolution();
        float rhsAfterSubtract = c - b;

        List<String> lines = new ArrayList<>();
        lines.add(formatEquation(a, b, c));
        if (b == 0f) {
            lines.add(formatXEquals(a, c, x));
        } else {
            lines.add(formatAxEquals(a, rhsAfterSubtract, b, c));
            lines.add(formatXEquals(a, rhsAfterSubtract, x));
        }
        lines.add(formatCheck(a, b, x, c));
        return lines;
    }

    private static String formatEquation(float a, float b, float c) {
        return formatCoeff(a) + "x" + formatOffset(b) + " = " + trim(c);
    }

    private static String formatAxEquals(float a, float rhs, float b, float c) {
        if (b == 0f) {
            return formatCoeff(a) + "x = " + trim(c);
        }
        StringBuilder sb = new StringBuilder();
        sb.append(formatCoeff(a)).append("x = ");
        sb.append(trim(c));
        if (b > 0) {
            sb.append(" - ").append(trim(b));
        } else {
            sb.append(" + ").append(trim(-b));
        }
        sb.append(" = ").append(trim(rhs));
        return sb.toString();
    }

    private static String formatXEquals(float a, float rhs, float x) {
        return "x = " + trim(rhs) + " / " + trim(a) + " = " + trim(x);
    }

    private static String formatCheck(float a, float b, float x, float c) {
        float left = a * x + b;
        return "check: " + formatCoeff(a) + "(" + trim(x) + ")" + formatOffset(b)
                + " = " + trim(left) + " = " + trim(c);
    }

    private static String formatCoeff(float a) {
        if (a == 1f) {
            return "";
        }
        if (a == -1f) {
            return "-";
        }
        return trim(a);
    }

    private static String formatOffset(float b) {
        if (b == 0f) {
            return "";
        }
        if (b > 0) {
            return " + " + trim(b);
        }
        return " - " + trim(-b);
    }

    private static String trim(float value) {
        if (value == (long) value) {
            return Long.toString((long) value);
        }
        return String.format(Locale.US, "%.4g", value);
    }
}
