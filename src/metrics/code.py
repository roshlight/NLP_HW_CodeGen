import ast


def syntax_valid(code):
    try:
        tree = ast.parse(code)
        return any(isinstance(n, ast.FunctionDef) and n.name == "solve" for n in tree.body)
    except (SyntaxError, ValueError, RecursionError):
        return False


def generation_metrics(predictions, references):
    n = max(1, len(predictions))
    return {
        "exact_match": sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / n,
        "syntax_valid": sum(map(syntax_valid, predictions)) / n,
    }
