"""
SyGMa comes currently with two rulesets:

phase1
    Phase 1 metabolism rules include mainly different types of oxidation, hydrolysis, reduction
    and condensation reactions

phase2
    Phase 2 metabolism rules include severaly conjugation reaction,
    i.e. with glucuronyl, sulfate, methyl and acetyl
"""
import os

# Locate the bundled rule files relative to this file so SyGMa works without
# setuptools/pkg_resources (which isn't present in lean/newer Python envs).
_RULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")

ruleset = {
    "phase1": os.path.join(_RULES_DIR, "phase1.txt"),
    "phase2": os.path.join(_RULES_DIR, "phase2.txt"),
}
