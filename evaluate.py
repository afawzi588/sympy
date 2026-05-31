"""
Evaluator for SymPy's factorint function.
Tests 1000 random (but deterministic) numbers of varying difficulty.
"""

import json
import os
import random
import signal
import sys
from contextlib import redirect_stdout
from typing import Tuple, Optional, cast

from sympy.core.random import seed as sympy_seed
from sympy.ntheory import factorint, isprime, randprime


class TimeoutError(Exception):
    """Raised when a factorization times out."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Factorization timed out")


def factorint_with_timeout(n: int, timeout_seconds: float = 1.0) -> Tuple[str, object]:
    """
    Run factorint with a timeout, without propagating exceptions.

    Args:
        n: Number to factor
        timeout_seconds: Maximum time allowed for factorization

    Returns:
        (status, payload):
            ("ok", factors)   - factors is the dict of prime factors
            ("timeout", None) - factorization exceeded timeout_seconds
            ("error", msg)    - factorint raised; msg is the error string

    The SIGALRM handler raises internally to interrupt the running call, but
    that is fully contained here and converted to a status; nothing is raised
    to the caller.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return "ok", factorint(n)
    except TimeoutError:
        return "timeout", None
    except Exception as e:  # noqa: BLE001 - any factorint failure counts as a failed case
        return "error", str(e)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


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


def generate_test_numbers(seed: int = 42, count: int = 100) -> list:
    """
    Generate a mix of easy and challenging numbers for factorization.
    Uses a fixed seed for reproducibility.
    Designed so baseline algorithm achieves ~50% success rate with 1s timeout.
    Hard numbers take ~5 seconds to factor (just beyond the 1s limit).
    """
    # Seed BOTH the global RNG (used by random.randint/choice below) and
    # SymPy's private RNG (used internally by randprime). Without the second
    # seed, randprime draws different primes every run, so the semiprime test
    # set (and thus the fitness) would not be reproducible.
    random.seed(seed)
    sympy_seed(seed)
    numbers = []

    # === EASY (should pass with 1s timeout) - 50 numbers ===

    # Category 1: Small numbers - 15 numbers
    for _ in range(15):
        numbers.append(random.randint(2, 10000))

    # Category 2: Products of small primes - 15 numbers
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for _ in range(15):
        num_factors = random.randint(2, 10)
        n = 1
        for _ in range(num_factors):
            n *= random.choice(small_primes)
        numbers.append(n)

    # Category 3: Small semiprimes - 10 numbers
    for _ in range(10):
        p1 = randprime(10**3, 10**5)
        p2 = randprime(10**3, 10**5)
        numbers.append(p1 * p2)

    # Category 4: Medium semiprimes - 10 numbers
    for _ in range(10):
        p1 = randprime(10**6, 10**9)
        p2 = randprime(10**6, 10**9)
        numbers.append(p1 * p2)

    # === HARD (timeout with 1s, but solvable in ~5s) - 50 numbers ===
    # Semiprimes with primes around 10^20-10^21 take ~5-6 seconds

    # Category 5: Hard semiprimes (primes ~10^20) - 25 numbers
    for _ in range(25):
        p1 = randprime(10**20, 2*10**20)
        p2 = randprime(10**20, 2*10**20)
        numbers.append(p1 * p2)

    # Category 6: Harder semiprimes (primes ~10^21) - 25 numbers
    for _ in range(25):
        p1 = randprime(10**21, 2*10**21)
        p2 = randprime(10**21, 2*10**21)
        numbers.append(p1 * p2)

    return numbers[:count]


def run_evaluation() -> Tuple[bool, str, float]:
    """
    Run the factorint evaluation.

    Returns:
        (is_valid: bool, message: str, success_rate: float)
    """
    numbers = generate_test_numbers(seed=42, count=100)

    passed = 0
    failed = 0
    first_error = None

    timeout_seconds = 0.1

    for n in numbers:
        status, payload = factorint_with_timeout(n, timeout_seconds=timeout_seconds)
        if status == "ok":
            is_valid, error_msg = validate_factorization(n, cast(dict, payload))
            if is_valid:
                passed += 1
            else:
                failed += 1
                if first_error is None:
                    first_error = f"n={n}: {error_msg}"
        elif status == "timeout":
            failed += 1
            if first_error is None:
                first_error = f"n={n}: Factorization timed out (>{timeout_seconds}s)"
        else:  # status == "error"
            failed += 1
            if first_error is None:
                first_error = f"n={n}: Exception - {payload}"

    success_rate = passed / (passed + failed)

    if failed == 0:
        return True, "All factorizations are correct.", success_rate
    else:
        return False, first_error, success_rate


if __name__ == "__main__":
    # Run evaluation
    # with redirect_stdout(open(os.devnull, 'w')):
    is_valid, error_message, success_rate = run_evaluation()

    # Output the results (success_rate is the fitness, no error raised on failures)
    print(json.dumps({
        "output": {
            "fitness": success_rate,
            "signature": (success_rate,)
        },
        "metainfo": "Success" if is_valid else error_message
    }))
