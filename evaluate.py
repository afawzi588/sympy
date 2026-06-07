"""
Evaluator for SymPy's factorint function.

Fitness = -(average number of operations) needed to correctly factor a fixed,
deterministic set of numbers of varying difficulty. The search should therefore
drive toward algorithms that factor the same numbers in *fewer operations*.

What "operations" means, and why it's deterministic
---------------------------------------------------
We define one operation as one executed Python line, counted with
``sys.settrace``. This is fully reproducible: ``factorint`` (and the rho / p-1 /
ECM / QS helpers it calls) seed their own local RNGs from explicit arguments and
never touch SymPy's global RNG, so the exact same sequence of lines runs every
time for a given input -- the same number always costs the same count.

This replaces the old wall-clock SIGALRM timeout, which was inherently
non-reproducible: whether a number finished within the time limit depended on
CPU speed and system load, so the score drifted between runs.

The counter lives here in the evaluator, never in the factoring code, so it
keeps measuring honestly no matter how the target files (factor_.py / ecm.py /
qs.py) are rewritten.

Correctness gate
----------------
Minimizing operations *without* checking correctness has a degenerate optimum:
returning ``{}`` costs almost nothing and would look like the best algorithm
ever. So any call that returns a wrong/incomplete factorization -- or fails to
finish within MAX_LINES, or raises -- is charged the full MAX_LINES penalty.
Only a verified-correct factorization is charged its actual line count.

The test numbers are all comfortably factorable by the baseline well under
MAX_LINES, so the cap only ever penalizes regressions/bugs; it is not a binding
limit for a correct algorithm. (A pathologically slow candidate can hit the cap
on many numbers and run longer than ~10s, but it is bounded by the sandbox
evaluation_timeout and gets a bad score anyway.)
"""

import json
import random
import sys
from typing import Tuple, Optional, cast

from sympy.core.random import seed as sympy_seed
from sympy.ntheory import factorint, isprime, randprime


# Per-number operation cap (executed Python lines). Set well above the cost of
# the hardest *correctly* factored test number under the baseline, so a correct
# algorithm never hits it -- it only bounds the cost charged to wrong, timed-out,
# or crashing candidates, and keeps total runtime bounded.
MAX_LINES = 8_000_000


class _BudgetExceeded(Exception):
    """Raised from the trace hook once the line budget is spent."""
    pass


def factorint_with_budget(n: int, max_lines: int = MAX_LINES) -> Tuple[str, object, int]:
    """
    Run factorint while deterministically counting executed Python lines.

    Args:
        n: Number to factor
        max_lines: Operation cap; the call is aborted once it is exceeded

    Returns:
        (status, payload, operations):
            ("ok", factors, k)   - completed in k operations; factors is the dict
            ("budget", None, k)  - exceeded max_lines (k == max_lines + 1)
            ("error", msg, k)    - factorint raised; msg is the error string

    A ``sys.settrace`` hook counts every executed line across all frames of the
    call and raises ``_BudgetExceeded`` the moment the count passes ``max_lines``.
    That exception is fully contained here and converted to a status. Because the
    abort point is fixed by the line count (not wall-clock time), the result is
    identical on every run.
    """
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line":
            count += 1
            if count > max_lines:
                raise _BudgetExceeded
        return tracer

    prev = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = factorint(n)
        return "ok", result, count
    except _BudgetExceeded:
        return "budget", None, count
    except Exception as e:  # noqa: BLE001 - any factorint failure counts as a failed case
        return "error", str(e), count
    finally:
        sys.settrace(prev)


def validate_factorization(n: int, factors: dict) -> Tuple[bool, str]:
    """
    Validates that a factorization is correct.

    Args:
        n: The number that was factored
        factors: Dictionary with prime factors as keys and multiplicities as values

    Returns:
        (is_valid: bool, message: str)
    """
    msg = "The factorization is correct."

    if n == 0:
        if factors == {0: 1}:
            return True, msg
        return False, f"Expected {{0: 1}} for n=0, got {factors}"

    if n == 1:
        if factors == {}:
            return True, msg
        return False, f"Expected {{}} for n=1, got {factors}"

    # Check that all factors are prime
    for p in factors:
        if p == -1:
            continue
        if not isprime(p):
            return False, f"Factor {p} is not prime"

    # Check that the product equals n
    product = 1
    for p, exp in factors.items():
        product *= p ** exp

    if product != n:
        return False, f"Product of factors ({product}) does not equal n ({n})"

    return True, msg


def generate_test_numbers(seed: int = 42) -> list:
    """
    Generate a deterministic mix of numbers spanning a range of difficulty,
    all of which the baseline can factor well within MAX_LINES operations.

    The spread (cheap trial-division cases up through ~10^10 semiprimes that
    cost a couple million operations) is what gives the "average operations"
    objective a smooth gradient: a better algorithm shaves operations off the
    harder cases and the average drops.

    Uses fixed seeds so the set -- and therefore every operation count -- is
    reproducible. Both the global RNG (random.*) and SymPy's private RNG (used
    by randprime) are seeded.
    """
    random.seed(seed)
    sympy_seed(seed)
    numbers = []

    # Category 1: small numbers - 15 (factored by trial division, ~tens of ops)
    for _ in range(15):
        numbers.append(random.randint(2, 10000))

    # Category 2: products of small primes - 10
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for _ in range(10):
        num_factors = random.randint(2, 10)
        n = 1
        for _ in range(num_factors):
            n *= random.choice(small_primes)
        numbers.append(n)

    # Category 3: small semiprimes (primes 10^3-10^6) - 10
    for _ in range(10):
        p1 = randprime(10**3, 10**6)
        p2 = randprime(10**3, 10**6)
        numbers.append(p1 * p2)

    # Category 4: medium semiprimes (primes 10^7-10^9) - 12
    for _ in range(12):
        p1 = randprime(10**7, 10**9)
        p2 = randprime(10**7, 10**9)
        numbers.append(p1 * p2)

    # Category 5: harder semiprimes (primes ~10^10) - 8
    # These dominate the cost (~1-3M operations each) and carry most of the
    # optimization gradient, while still finishing far below MAX_LINES.
    for _ in range(8):
        p1 = randprime(10**10, 2 * 10**10)
        p2 = randprime(10**10, 2 * 10**10)
        numbers.append(p1 * p2)

    return numbers


def run_evaluation() -> Tuple[float, int, int, Optional[str]]:
    """
    Run the factorint evaluation.

    Returns:
        (avg_operations, solved, total, first_error)
            avg_operations - mean operation count, charging MAX_LINES for any
                             number that was not factored correctly
            solved         - count of numbers factored correctly within budget
            total          - number of test numbers
            first_error    - description of the first failure, or None
    """
    numbers = generate_test_numbers(seed=42)

    costs = []
    solved = 0
    first_error = None

    for n in numbers:
        status, payload, operations = factorint_with_budget(n)

        if status == "ok":
            is_valid, error_msg = validate_factorization(n, cast(dict, payload))
            if is_valid:
                costs.append(operations)
                solved += 1
                continue
            failure = f"n={n}: {error_msg}"
        elif status == "budget":
            failure = f"n={n}: exceeded operation budget (>{MAX_LINES} lines)"
        else:  # status == "error"
            failure = f"n={n}: Exception - {payload}"

        # Correctness gate: anything not verified-correct pays the full cap.
        costs.append(MAX_LINES)
        if first_error is None:
            first_error = failure

    avg_operations = sum(costs) / len(costs)
    return avg_operations, solved, len(numbers), first_error


if __name__ == "__main__":
    avg_operations, solved, total, first_error = run_evaluation()

    # Fitness: higher is better, so we negate the average operation count
    # (fewer operations -> larger fitness). The raw average and the
    # solved/total tally are reported in metainfo for visibility.
    fitness = -avg_operations
    metainfo = (
        f"avg_operations={avg_operations:.1f}, solved={solved}/{total}"
        + ("" if first_error is None else f", first_error: {first_error}")
    )
    print(json.dumps({
        "output": {
            "fitness": fitness,
            "signature": (fitness,)
        },
        "metainfo": "Success"
    }))
