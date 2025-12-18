import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import params
import ring
import sampling


@dataclass
class PKEPublicKey:
    seedA: bytes
    b: np.ndarray  # shape (k, N)


@dataclass
class PKESecretKey:
    s: np.ndarray  # shape (k, N)


def split_seed(seed: bytes) -> Tuple[bytes, bytes, bytes]:
    expanded = hashlib.shake_128(seed).digest(3 * params.SEED_BYTES)
    return (
        expanded[: params.SEED_BYTES],
        expanded[params.SEED_BYTES : 2 * params.SEED_BYTES],
        expanded[2 * params.SEED_BYTES :],
    )


def keygen(seed: Optional[bytes] = None) -> Tuple[PKEPublicKey, PKESecretKey]:
    if seed is None:
        seed = sampling.random_seed()
    seedA, seedsk, seede = split_seed(seed)

    A = sampling.expand_matrix(seedA)
    s = sampling.sample_hwt(seedsk, params.H_S)
    e = sampling.sample_discrete_gaussian(seede, params.SIGMA)

    # b = -A^T * s + e mod q
    A_T = np.transpose(A, (1, 0, 2))
    As = ring.mat_vec_mul(A_T, s, q=params.Q)
    b = (e - As) % params.Q

    return PKEPublicKey(seedA=seedA, b=b), PKESecretKey(s=s)

def encode_message_poly(msg_poly: np.ndarray) -> np.ndarray:
    """Ensure message coefficients are in [0, T)."""
    msg_poly = np.asarray(msg_poly, dtype=params.DTYPE)
    if msg_poly.shape != (params.N,):
        raise ValueError(f"Message poly must have length N={params.N}")
    return msg_poly % params.T


def bytes_to_poly(msg: bytes) -> np.ndarray:
    msg = bytes(msg)
    coeffs = np.zeros(params.N, dtype=params.DTYPE)
    if params.T == 2:
        if len(msg) != params.MU_BYTES:
            raise ValueError(f"Expected {params.MU_BYTES} bytes for µ (t=2).")
        for i in range(params.N):
            coeffs[i] = (msg[i // 8] >> (i & 7)) & 1
    else:
        if len(msg) > params.N:
            raise ValueError(f"Message too long for polynomial encoding (max {params.N} bytes)")
        for i, b in enumerate(msg):
            coeffs[i] = b % params.T
    return coeffs


def poly_to_bytes(poly: np.ndarray, out_len: int) -> bytes:
    poly = np.asarray(poly, dtype=params.DTYPE)
    if params.T == 2:
        if out_len != params.MU_BYTES:
            raise ValueError(f"Expected {params.MU_BYTES} bytes for µ (t=2).")
        out = bytearray(out_len)
        for i in range(params.N):
            out[i // 8] |= (int(poly[i] & 1) << (i & 7))
        return bytes(out)

    if out_len > params.N:
        raise ValueError(f"Requested {out_len} bytes, but only {params.N} coefficients available")
    return bytes(int(poly[i] % params.T) for i in range(out_len))


def encrypt(pk: PKEPublicKey, msg_poly: np.ndarray, seedr: Optional[bytes] = None) -> Tuple[np.ndarray, np.ndarray]:
    A = sampling.expand_matrix(pk.seedA)
    if seedr is None:
        seedr = sampling.random_seed()
    r = sampling.sample_hwt(seedr, params.H_R)

    msg_poly = encode_message_poly(msg_poly)

    Ar = ring.mat_vec_mul(A, r, q=params.Q)
    c1 = ring.compress(Ar, q=params.Q, p=params.P)

    inner = ring.vec_dot(pk.b, r, q=params.Q)
    term1 = ring.compress(inner, q=params.Q, p=params.P_PRIME)
    term2 = (msg_poly * (params.P_PRIME // params.T)) % params.P_PRIME
    c2 = (term1 + term2) % params.P_PRIME
    return c1, c2


def decrypt(sk: PKESecretKey, ct: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    c1, c2 = ct
    # Spec: µ' = round(t/p * <c1,s> + t/p' * c2) mod t
    u = ring.vec_dot(c1, sk.s, q=params.P)  # element in R_p

    def centered(x: np.ndarray, mod: int) -> np.ndarray:
        x = (np.asarray(x, dtype=params.DTYPE) % mod).astype(params.DTYPE)
        half = mod // 2
        return np.where(x >= half, x - mod, x)

    def round_div_pow2(x: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0:
            return x.astype(params.DTYPE)
        half = 1 << (shift - 1)
        x = x.astype(params.DTYPE)
        pos = x >= 0
        out = np.empty_like(x, dtype=params.DTYPE)
        out[pos] = (x[pos] + half) >> shift
        out[~pos] = -(((-x[~pos]) + half) >> shift)
        return out

    u_c = centered(u, params.P)
    c2_c = centered(c2, params.P_PRIME)

    shift_p = (params.P.bit_length() - 1) - (params.T.bit_length() - 1)
    shift_pprime = (params.P_PRIME.bit_length() - 1) - (params.T.bit_length() - 1)
    shift = max(shift_p, shift_pprime)

    sum_scaled = (u_c << (shift - shift_p)) + (c2_c << (shift - shift_pprime))
    msg = round_div_pow2(sum_scaled, shift) % params.T
    return msg


def encrypt_bytes(pk: PKEPublicKey, msg: bytes, seedr: Optional[bytes] = None) -> Tuple[np.ndarray, np.ndarray]:
    msg_poly = bytes_to_poly(msg)
    return encrypt(pk, msg_poly, seedr=seedr)


def decrypt_to_bytes(sk: PKESecretKey, ct: Tuple[np.ndarray, np.ndarray], out_len: int) -> bytes:
    poly = decrypt(sk, ct)
    return poly_to_bytes(poly, out_len)
