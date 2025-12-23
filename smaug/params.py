from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@dataclass(frozen=True)
class SmaugParamSet:
    name: str
    n: int
    k: int
    q: int
    p: int
    p_prime: int
    t: int
    hs: int
    hr: int
    sigma: float


PARAM_SETS: dict[str, SmaugParamSet] = {
    "SMAUG-128": SmaugParamSet(
        name="SMAUG-128",
        n=256,
        k=2,
        q=1024,
        p=256,
        p_prime=32,
        t=2,
        hs=140,
        hr=132,
        sigma=1.0625,
    ),
    "SMAUG-192": SmaugParamSet(
        name="SMAUG-192",
        n=256,
        k=3,
        q=2048,
        p=256,
        p_prime=256,
        t=2,
        hs=198,
        hr=151,
        sigma=1.453713,
    ),
    "SMAUG-256": SmaugParamSet(
        name="SMAUG-256",
        n=256,
        k=5,
        q=2048,
        p=256,
        p_prime=64,
        t=2,
        hs=176,
        hr=160,
        sigma=1.0625,
    ),
    # Parameter sets matching the provided reference KAT byte sizes.
    # Note: These differ from Table 3 in the spec (notably q and p').
    "SMAUG-128-KAT": SmaugParamSet(
        name="SMAUG-128-KAT",
        n=256,
        k=2,
        q=1024,
        p=256,
        p_prime=256,
        t=2,
        hs=140,
        hr=132,
        sigma=1.0625,
    ),
    "SMAUG-192-KAT": SmaugParamSet(
        name="SMAUG-192-KAT",
        n=256,
        k=3,
        q=1024,
        p=256,
        p_prime=256,
        t=2,
        hs=150,
        hr=147,
        sigma=1.453713,
    ),
    "SMAUG-256-KAT": SmaugParamSet(
        name="SMAUG-256-KAT",
        n=256,
        k=5,
        q=1024,
        p=256,
        p_prime=256,
        t=2,
        hs=145,
        hr=140,
        sigma=1.0625,
    ),
    "TOY-NOISELESS": SmaugParamSet(
        name="TOY-NOISELESS",
        n=256,
        k=2,
        q=256,
        p=256,
        p_prime=256,
        t=2,
        hs=32,
        hr=32,
        sigma=0.0,
    ),
}


# Byte lengths used by the spec
SEED_BYTES = 32
HASH_BYTES = 32
MU_BYTES = 32  # 256-bit message µ for KEM

DTYPE = np.int64


def set_active(name: str) -> SmaugParamSet:
    """
    Select the active parameter set and update module-level derived constants.
    """
    global ACTIVE, N, K, Q, P, P_PRIME, T, H_S, H_R, SIGMA
    par = PARAM_SETS[name]

    if not (_is_power_of_two(par.q) and _is_power_of_two(par.p) and _is_power_of_two(par.p_prime) and _is_power_of_two(par.t)):
        raise ValueError("This implementation assumes power-of-two moduli (q, p, p', t).")
    if par.q % par.p != 0 or par.q % par.p_prime != 0:
        raise ValueError("Expected p | q and p' | q.")
    if par.p % par.t != 0 or par.p_prime % par.t != 0:
        raise ValueError("Expected t | p and t | p'.")
    if par.n != 8 * MU_BYTES and par.t == 2:
        raise ValueError("For t=2, this code expects n == 256 to carry µ bits.")

    ACTIVE = par
    N = par.n
    K = par.k
    Q = par.q
    P = par.p
    P_PRIME = par.p_prime
    T = par.t
    H_S = par.hs
    H_R = par.hr
    SIGMA = par.sigma
    return par


# Default to a real parameter set (can be changed by the caller).
ACTIVE: SmaugParamSet
N: int
K: int
Q: int
P: int
P_PRIME: int
T: int
H_S: int
H_R: int
SIGMA: float
set_active("SMAUG-128")
