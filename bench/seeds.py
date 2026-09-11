"""Disjoint seed sets.

Calibration, development, and evaluation seeds never overlap. Calibration seeds are the only
ones a tolerance may be tuned on; development seeds are for building and debugging; and
evaluation seeds are held out until the policy is frozen.

This is bookkeeping, not proof of generalization. Held-out seeds give held-out *executions of
known fault families*. They say nothing about unknown real-world bugs, and no report may
present them as if they did.
"""

from __future__ import annotations

CALIBRATION_SEEDS: tuple[int, ...] = tuple(range(1_000, 1_016))
DEVELOPMENT_SEEDS: tuple[int, ...] = tuple(range(2_000, 2_016))
EVALUATION_SEEDS: tuple[int, ...] = tuple(range(3_000, 3_016))

SEED_SETS: dict[str, tuple[int, ...]] = {
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "evaluation": EVALUATION_SEEDS,
}


def assert_disjoint() -> None:
    """Raise if any seed appears in more than one set."""
    seen: dict[int, str] = {}
    for name, seeds in SEED_SETS.items():
        if len(set(seeds)) != len(seeds):
            raise AssertionError(f"seed set {name!r} contains duplicates")
        for seed in seeds:
            if seed in seen:
                raise AssertionError(f"seed {seed} appears in both {seen[seed]!r} and {name!r}")
            seen[seed] = name


assert_disjoint()

__all__ = [
    "CALIBRATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "EVALUATION_SEEDS",
    "SEED_SETS",
    "assert_disjoint",
]
