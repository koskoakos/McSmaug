import hashlib

import numpy as np

import params
import rng


def xof128(seed: bytes, outlen: int) -> bytes:
    return hashlib.shake_128(seed).digest(outlen)


def shake256(seed: bytes, outlen: int) -> bytes:
    return hashlib.shake_256(seed).digest(outlen)


def seed_to_rng(seed: bytes) -> np.random.Generator:
    # Derive 256 bits for numpy seed
    seed_bytes = xof128(seed, 32)
    seed_int = int.from_bytes(seed_bytes, "big")
    return np.random.default_rng(seed_int)


def _bytes_to_poly_q(buf: bytes, q: int, n: int) -> np.ndarray:
    """
    Parse a bitstream into n coefficients in [0, q) assuming q is power-of-two.
    Coefficients are read little-endian, LSB-first.
    """
    logq = (q.bit_length() - 1)
    mask = q - 1
    total_bits = n * logq
    total_bytes = (total_bits + 7) // 8
    if len(buf) < total_bytes:
        raise ValueError("Insufficient bytes for polynomial expansion.")

    out = np.zeros(n, dtype=params.DTYPE)
    acc = 0
    acc_bits = 0
    byte_idx = 0
    for i in range(n):
        while acc_bits < logq:
            acc |= buf[byte_idx] << acc_bits
            acc_bits += 8
            byte_idx += 1
        out[i] = acc & mask
        acc >>= logq
        acc_bits -= logq
    return out


def expand_matrix(seed: bytes) -> np.ndarray:
    """
    Deterministically expand seed into uniform matrix A in R_q^{k x k}.
    """
    q = params.Q
    n = params.N
    k = params.K

    logq = q.bit_length() - 1
    poly_bits = n * logq
    poly_bytes = (poly_bits + 7) // 8
    buf = xof128(seed, poly_bytes * k * k)

    A = np.zeros((k, k, n), dtype=params.DTYPE)
    for idx in range(k * k):
        start = idx * poly_bytes
        poly = _bytes_to_poly_q(buf[start : start + poly_bytes], q=q, n=n)
        i, j = divmod(idx, k)
        A[i, j] = poly
    return A


def sample_hwt(seed: bytes, weight: int) -> np.ndarray:
    """
    Sample a sparse ternary vector (length k of polynomials) with total Hamming weight.
    Signs are uniform over {+1, -1}.
    """
    rng = seed_to_rng(seed)
    out = np.zeros((params.K, params.N), dtype=params.DTYPE)
    total = params.K * params.N
    if weight > total:
        raise ValueError("Hamming weight exceeds vector size.")
    positions = rng.choice(total, size=weight, replace=False)
    signs = rng.choice(np.array([1, -1], dtype=params.DTYPE), size=weight, replace=True)
    poly_idx = positions // params.N
    coeff_idx = positions % params.N
    out[poly_idx, coeff_idx] = signs
    return out


def sample_discrete_gaussian(seed: bytes, sigma: float) -> np.ndarray:
    if sigma == 0:
        return np.zeros((params.K, params.N), dtype=params.DTYPE)
    rng = seed_to_rng(seed)
    samples = rng.normal(loc=0.0, scale=sigma, size=(params.K, params.N))
    return samples.round().astype(params.DTYPE)


def random_bytes(n: int) -> bytes:
    return rng.random_bytes(n)


def random_seed() -> bytes:
    return random_bytes(params.SEED_BYTES)
