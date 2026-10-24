from src.metrics.arithmetic import arithmetic_value, arithmetic_metrics


def test_arithmetic():
    assert (
        arithmetic_value(
            "def solve():\n    x = 3\n    for i in range(4):\n        x += i\n    return x\n"
        )
        == 9
    )
    assert arithmetic_value("import math\ndef solve():\n    return math.sqrt(9)\n") == 3


def test_no_python_execution_or_unbounded_work():
    assert arithmetic_value('def solve():\n    return __import__("os").system("echo bad")') is None
    assert arithmetic_value("def solve():\n    return 10 ** 1000000") is None
    assert arithmetic_value("def solve():\n    return sum(range(1000000000))") is None
    assert arithmetic_value("def solve():\n    while True:\n        pass") is None


def test_reference_coverage_is_explicit():
    m = arithmetic_metrics(
        ["def solve():\n    return 3"] * 2,
        ["def solve():\n    return 3", "def solve():\n    return unknown()"],
    )
    assert m["arithmetic_reference_coverage"] == 0.5
    assert m["arithmetic_agreement_supported"] == 1
    assert m["arithmetic_agreement_all"] == 0.5
