"""The sudoers entry documented for the GPU clock lock must only allow the
two argument lists the engine sends: `-lgc <floor>,<ceiling>` and `-rgc`.

A `-lgc *` wildcard also matches spaces, so it let any program running as
the user pass extra nvidia-smi options as root without a password.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = [*sorted((REPO / "docs").glob("*.md")), REPO / "README.md"]


def _rules() -> list[str]:
    return [
        line.split("/usr/bin/nvidia-smi", 1)[1].strip()
        for doc in DOCS
        for line in doc.read_text().splitlines()
        if "NOPASSWD:" in line and "/usr/bin/nvidia-smi" in line
    ]


def _allows(rule: str, args: str) -> bool:
    """sudoers argument matching for the two forms used here: an anchored
    regex (sudo >= 1.9.10) or a literal argument list."""
    if rule.startswith("^") and rule.endswith("$"):
        return re.fullmatch(rule[1:-1], args) is not None
    assert "*" not in rule and "?" not in rule, f"wildcard in sudoers rule: {rule!r}"
    return rule == args


def test_documented_rules_allow_exactly_what_the_engine_sends() -> None:
    rules = _rules()
    assert rules, "the clock-lock sudoers rules are no longer documented"

    def allowed(args: str) -> bool:
        return any(_allows(r, args) for r in rules)

    assert allowed("-lgc 1845,2100")
    assert allowed("-rgc")
    for evil in (
        "-lgc 1,1 -f /etc/ld.so.preload",
        "-lgc 1,1 -r",
        "-lgc --filename=/etc/x",
        "-rgc -f /etc/x",
        "-pl 1",
    ):
        assert not allowed(evil), evil
