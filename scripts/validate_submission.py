"""Validate the bot format and prove correspondence with the original questions."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def validate(questions_path, submission_path, details_path=None):
    questions_path, submission_path = Path(questions_path), Path(submission_path)
    questions = [json.loads(s) for s in questions_path.read_text(encoding="utf-8").splitlines()]
    answers = [json.loads(s) for s in submission_path.read_text(encoding="utf-8").splitlines()]
    if len(questions) != 1919 or len(answers) != 1919:
        raise ValueError(f"Expected 1919 questions and answers: {len(questions)}, {len(answers)}")
    for i, answer in enumerate(answers):
        if set(answer) != {"code"} or not isinstance(answer["code"], str):
            raise ValueError(f"Invalid answer schema at row {i + 1}")
    if details_path is not None:
        details = [json.loads(s) for s in Path(details_path).read_text(encoding="utf-8").splitlines()]
        if len(details) != 1919:
            raise ValueError("Incomplete inference details")
        for i, (question, answer, detail) in enumerate(zip(questions, answers, details)):
            if detail["id"] != i or detail["question"] != question["question"] or detail["code"] != answer["code"]:
                raise ValueError(f"Question/answer order differs at row {i + 1}")
    syntax = 0
    for answer in answers:
        try:
            ast.parse(answer["code"])
            syntax += bool(answer["code"].strip())
        except (SyntaxError, ValueError, RecursionError):
            pass
    return {
        "rows": len(answers),
        "empty_outputs": sum(not a["code"].strip() for a in answers),
        "syntax_valid": syntax,
        "order_verified": details_path is not None,
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        "official_pass_at_1": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--submission", default="submission.jsonl")
    parser.add_argument("--details")
    parser.add_argument("--output", default="artifacts/submission_validation.json")
    args = parser.parse_args()
    result = validate(args.questions, args.submission, args.details)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
