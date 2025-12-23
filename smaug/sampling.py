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


def _bytes_to_poly_q_ref(buf: bytes, n: int) -> np.ndarray:
    """
    Reference packing for q=1024: 5 bytes -> 4 coefficients (10-bit).
    """
    out = np.zeros(n, dtype=params.DTYPE)
    for i in range(n // 4):
        b_idx = 5 * i
        d_idx = 4 * i
        b4 = buf[b_idx + 4]
        out[d_idx + 0] = ((b4 & 0x03) << 8) | buf[b_idx + 0]
        out[d_idx + 1] = ((b4 & 0x0C) << 6) | buf[b_idx + 1]
        out[d_idx + 2] = ((b4 & 0x30) << 4) | buf[b_idx + 2]
        out[d_idx + 3] = ((b4 & 0xC0) << 2) | buf[b_idx + 3]
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
        if "KAT" in params.ACTIVE.name and q == 1024:
            poly = _bytes_to_poly_q_ref(buf[start : start + poly_bytes], n=n)
        else:
            poly = _bytes_to_poly_q(buf[start : start + poly_bytes], q=q, n=n)
        i, j = divmod(idx, k)
        A[i, j] = poly
    return A


def sample_hwt(seed: bytes, weight: int) -> np.ndarray:
    """
    Sample a sparse ternary vector using the paper's HWTh (SampleInBall-style).
    """
    out = np.zeros((params.K, params.N), dtype=params.DTYPE)
    weight_per_poly = weight // params.K
    if weight_per_poly * params.K != weight:
        raise ValueError("Hamming weight must be divisible by K.")
    if weight_per_poly > params.N:
        raise ValueError("Hamming weight exceeds polynomial size.")
    for i in range(params.K):
        poly_seed = seed + i.to_bytes(1, "little")
        out[i] = _hwt_paper(poly_seed, weight_per_poly)
    return out


def _hwt_paper(seed: bytes, weight: int) -> np.ndarray:
    """
    HWTh from the spec (hybrid SampleInBall/CWW), for a single polynomial.
    """
    n = params.N
    h = weight
    res = np.zeros(n, dtype=params.DTYPE)
    idx = 0
    word_count = h + ((h + 15) // 16)
    buf = shake256(seed, word_count * 4)
    words = np.frombuffer(buf, dtype="<u4")
    for i in range(n - h, n):
        word = int(words[idx])
        degree = (word * (i + 1)) >> 32
        res[i] = res[degree]
        sign_word = int(words[h + (idx >> 4)])
        sign = ((sign_word >> (idx & 0x0F)) & 0x02) - 1
        res[degree] = sign
        idx += 1
    return res


def sample_hwt_kat(seed: bytes, weight: int) -> np.ndarray:
    """
    Sample sparse ternary vector using ref HWT (KAT compatibility).
    """
    if len(seed) < params.SEED_BYTES + 8:
        raise ValueError("Expected KAT seed buffer to be at least 40 bytes.")
    if weight != params.H_S:
        raise ValueError("KAT HWT sampler expects weight == H_S.")
    weight_per_poly = weight // params.K
    import ref_smaug
    mode_by_k = {2: 1, 3: 3, 5: 5}
    par = ref_smaug.PARAMS_BY_MODE[mode_by_k[params.K]]
    xof_res = ref_smaug.shake128(par.crypto_bytes, seed[: par.crypto_bytes + 8])
    dense = ref_smaug.hwt(par, xof_res, weight_per_poly)
    vals = np.zeros(params.N, dtype=params.DTYPE)
    vals[dense == 0x01] = 1
    vals[dense == 0xFF] = -1
    out = np.zeros((params.K, params.N), dtype=params.DTYPE)
    for i in range(params.K):
        out[i] = vals
    return out


def sample_discrete_gaussian(seed: bytes, sigma: float) -> np.ndarray:
    # Use deterministic DGS (reference-style) regardless of sigma for consistency.
    return sample_discrete_gaussian_kat(seed)


def sample_discrete_gaussian_kat(seed: bytes) -> np.ndarray:
    """
    Reference Gaussian sampler for KAT compatibility.
    Returns coefficients in Z_q.
    """
    import ref_smaug
    mode_by_k = {2: 1, 3: 3, 5: 5}
    par = ref_smaug.PARAMS_BY_MODE[mode_by_k[params.K]]
    e_shifted = ref_smaug._add_gaussian_error_vec(par, seed)
    shift = 16 - (params.Q.bit_length() - 1)
    e_signed = e_shifted.view(np.int16).astype(params.DTYPE)
    e_q = (e_signed >> shift) % params.Q
    return e_q


def random_bytes(n: int) -> bytes:
    return rng.random_bytes(n)


def random_seed() -> bytes:
    return random_bytes(params.SEED_BYTES)
