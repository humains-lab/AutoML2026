from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipegenie.evolutionary._individual import Individual

def extract_parentheses(s: str):
    stack = []
    pairs = []

    for i, c in enumerate(s):
        if c == "(":
            stack.append(i)
        elif c == ")":
            start = stack.pop()
            pairs.append((start, i))

    results = []

    for start, end in pairs:
        content = s[start + 1:end]

        parts = []
        current = []
        depth = 0

        for ch in content:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)

        if current:
            parts.append("".join(current))

        results.append(parts)

    return results

def hp_numbers(ind: Individual) -> int:
    hps_by_algorithm = extract_parentheses(str(ind))
    hp_counter = 0
    for hps in hps_by_algorithm:
        hp_counter += len(hps)
    return hp_counter