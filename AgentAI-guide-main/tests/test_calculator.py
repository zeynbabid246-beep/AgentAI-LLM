"""Unittests for the AST-safe calculator (tools.safe_eval / calculator tool)."""

import math

import pytest

from tools import CalculatorError, calculator, safe_eval


class TestSafeEvalBasics:
    def test_simple_arithmetic(self):
        assert safe_eval("2 + 3") == 5.0
        assert safe_eval("10 - 4 * 2") == 2.0
        assert safe_eval("(2 + 3) * 4") == 20.0
        assert safe_eval("10 / 4") == 2.5

    def test_powers_and_modulo(self):
        assert safe_eval("2 ** 10") == 1024.0
        assert safe_eval("2 ^ 10") == 1024.0  # caret normalization
        assert safe_eval("10 % 3") == 1.0
        assert safe_eval("7 // 2") == 3.0

    def test_unary_and_constants(self):
        assert safe_eval("-5 + 3") == -2.0
        assert safe_eval("+7") == 7.0
        assert math.isclose(safe_eval("pi"), math.pi)
        assert math.isclose(safe_eval("2 * pi"), 2 * math.pi)
        assert math.isclose(safe_eval("e * 0"), 0.0)

    def test_functions(self):
        assert safe_eval("sqrt(144)") == 12.0
        assert math.isclose(safe_eval("ln(e)"), 1.0)
        assert safe_eval("abs(-8)") == 8.0
        assert safe_eval("round(3.6)") == 4.0
        assert math.isclose(safe_eval("sin(0)"), 0.0)

    def test_float_result(self):
        assert safe_eval("sqrt(2)") == pytest.approx(1.4142135623730951)


class TestSafeEvalRejections:
    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('ls')",
            "().__class__.__bases__",
            "[].append(1)",
            "'a' + 'b'",
            "x + 1",
            "lambda: 1",
            "1 if True else 2",
            "",
            "   ",
            "2 +",
            "sqrt()",
            "unknown_fn(3)",
            "9 ** 9 ** 9 ** 2",  # exponent guard
        ],
    )
    def test_unsafe_or_invalid_expressions_raise(self, expr):
        with pytest.raises(CalculatorError):
            safe_eval(expr)

    def test_division_by_zero(self):
        with pytest.raises(CalculatorError, match="Division by zero"):
            safe_eval("1 / 0")


class TestCalculatorTool:
    def test_integer_result_formatting(self):
        out = calculator.invoke({"expression": "1250 * 12 / 3"})
        assert out == "1250 * 12 / 3 = 5000"

    def test_float_result_formatting(self):
        out = calculator.invoke({"expression": "10 / 4"})
        assert out == "10 / 4 = 2.5"

    def test_error_returns_message_not_exception(self):
        out = calculator.invoke({"expression": "1 / 0"})
        assert out.startswith("Calculator error:")

    def test_tool_metadata(self):
        assert calculator.name == "calculator"
        assert "arithmetic" in calculator.description.lower()
