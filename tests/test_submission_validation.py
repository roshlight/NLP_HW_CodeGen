import json

import pytest

from scripts.validate_submission import validate


def test_original_order_is_verified_and_reordering_rejected(tmp_path):
    questions = [{"question": f"Question {i}"} for i in range(1919)]
    answers = [{"code": f"def solve():\n    return {i}"} for i in range(1919)]
    details = [{"id": i, **q, **a} for i, (q, a) in enumerate(zip(questions, answers))]
    def write(name, records):
        p = tmp_path / name
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        return p
    q = write("questions.jsonl", questions)
    a = write("submission.jsonl", answers)
    d = write("details.jsonl", details)
    assert validate(q, a, d)["order_verified"]
    details[0], details[1] = details[1], details[0]
    write("details.jsonl", details)
    with pytest.raises(ValueError, match="order differs"):
        validate(q, a, d)
    answers[0]["id"] = 0
    write("submission.jsonl", answers)
    with pytest.raises(ValueError, match="schema"):
        validate(q, a)
