"""Arithmetic factorization reasoning task generation.

The task observes input/output integers related by a hidden program.  A program
is a sequence of primitive multiplications by 2, 3, 5, or 7.  Internally we keep
the prime-exponent state as a stable representation while still exposing the
corresponding integer values for reporting and sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


PRIMES: tuple[int, ...] = (2, 3, 5, 7)
ANCHOR_PROGRAMS: tuple[tuple[int, ...], ...] = ((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3))


@dataclass(frozen=True)
class ArithmeticExample:
    support_x: int
    support_y: int
    query_x: int
    query_y: int
    support_x_exp: tuple[int, ...]
    support_y_exp: tuple[int, ...]
    query_x_exp: tuple[int, ...]
    query_y_exp: tuple[int, ...]
    program: tuple[int, ...]


def enumerate_programs(
    *,
    min_len: int = 1,
    max_len: int = 3,
    num_actions: int = len(PRIMES),
) -> list[tuple[int, ...]]:
    if min_len < 0 or max_len < min_len:
        raise ValueError("expected 0 <= min_len <= max_len")
    frontier: list[tuple[int, ...]] = [()]
    programs: list[tuple[int, ...]] = []
    for length in range(1, max_len + 1):
        frontier = [program + (action,) for program in frontier for action in range(num_actions)]
        if length >= min_len:
            programs.extend(frontier)
    return programs


def apply_factor_program(value: int, program: tuple[int, ...]) -> int:
    result = value
    for action in program:
        result *= PRIMES[action]
    return result


def apply_exponent_program(exponents: tuple[int, ...], program: tuple[int, ...]) -> tuple[int, ...]:
    updated = list(exponents)
    for action in program:
        updated[action] += 1
    return tuple(updated)


def exponents_to_value(exponents: tuple[int, ...]) -> int:
    value = 1
    for prime, exponent in zip(PRIMES, exponents):
        value *= prime**exponent
    return value


def value_to_exponents(value: int) -> tuple[int, ...]:
    if value < 1:
        raise ValueError("value must be positive")
    exponents: list[int] = []
    remainder = value
    for prime in PRIMES:
        exponent = 0
        while remainder % prime == 0:
            exponent += 1
            remainder //= prime
        exponents.append(exponent)
    if remainder != 1:
        raise ValueError(f"{value} has factors outside {PRIMES}")
    return tuple(exponents)


def split_programs(
    *,
    alpha: float = 1.0,
    min_len: int = 1,
    max_len: int = 3,
    seed: int = 0,
    use_anchors: bool = True,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    rng = Random(seed)
    all_programs = enumerate_programs(min_len=min_len, max_len=max_len)
    anchors = {program for program in ANCHOR_PROGRAMS if min_len <= len(program) <= max_len} if use_anchors else set()
    train: set[tuple[int, ...]] = set(anchors)

    for length in range(min_len, max_len + 1):
        candidates = [program for program in all_programs if len(program) == length and program not in anchors]
        rng.shuffle(candidates)
        keep = len(candidates) if alpha == 1.0 else max(1, round(alpha * len(candidates)))
        train.update(candidates[:keep])

    train_list = sorted(train, key=lambda program: (len(program), program))
    train_set = set(train_list)
    comp_list = [program for program in all_programs if program not in train_set]
    return train_list, comp_list


def sample_exponents(rng: Random, *, max_start_exp: int = 3) -> tuple[int, ...]:
    return tuple(rng.randint(0, max_start_exp) for _ in PRIMES)


def sample_example(
    rng: Random,
    programs: list[tuple[int, ...]],
    *,
    max_start_exp: int = 3,
) -> ArithmeticExample:
    if not programs:
        raise ValueError("programs must not be empty")
    program = rng.choice(programs)
    support_x_exp = sample_exponents(rng, max_start_exp=max_start_exp)
    query_x_exp = sample_exponents(rng, max_start_exp=max_start_exp)
    support_y_exp = apply_exponent_program(support_x_exp, program)
    query_y_exp = apply_exponent_program(query_x_exp, program)
    return ArithmeticExample(
        support_x=exponents_to_value(support_x_exp),
        support_y=exponents_to_value(support_y_exp),
        query_x=exponents_to_value(query_x_exp),
        query_y=exponents_to_value(query_y_exp),
        support_x_exp=support_x_exp,
        support_y_exp=support_y_exp,
        query_x_exp=query_x_exp,
        query_y_exp=query_y_exp,
        program=program,
    )


def generate_examples(
    *,
    n: int,
    split: str,
    seed: int,
    alpha: float = 1.0,
    train_min_len: int = 1,
    train_max_len: int = 3,
    length_ood_min_len: int = 4,
    length_ood_max_len: int = 6,
    split_seed: int = 0,
    use_anchors: bool = True,
    max_start_exp: int = 3,
) -> list[ArithmeticExample]:
    train_programs, comp_programs = split_programs(
        alpha=alpha,
        min_len=train_min_len,
        max_len=train_max_len,
        seed=split_seed,
        use_anchors=use_anchors,
    )
    if split in {"train", "id"}:
        programs = train_programs
    elif split == "comp_ood":
        programs = comp_programs
    elif split == "length_ood":
        programs = enumerate_programs(min_len=length_ood_min_len, max_len=length_ood_max_len)
    else:
        raise ValueError(f"unknown split: {split}")

    rng = Random(seed)
    return [sample_example(rng, programs, max_start_exp=max_start_exp) for _ in range(n)]
