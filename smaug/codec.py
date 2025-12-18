from __future__ import annotations

import numpy as np

import params


def pack_bits_le(values: np.ndarray, bits: int) -> bytes:
    """
    Pack unsigned integers (LSB-first) into a byte string.
    """
    if bits <= 0:
        raise ValueError("bits must be > 0")
    values = np.asarray(values, dtype=np.int64).ravel()
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    out = bytearray()
    for v in values:
        acc |= (int(v) & mask) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_bits_le(buf: bytes, count: int, bits: int) -> np.ndarray:
    if bits <= 0:
        raise ValueError("bits must be > 0")
    mask = (1 << bits) - 1
    out = np.zeros(count, dtype=params.DTYPE)
    acc = 0
    acc_bits = 0
    idx = 0
    for i in range(count):
        while acc_bits < bits:
            if idx >= len(buf):
                raise ValueError("Insufficient bytes to unpack")
            acc |= buf[idx] << acc_bits
            acc_bits += 8
            idx += 1
        out[i] = acc & mask
        acc >>= bits
        acc_bits -= bits
    return out


def pack_poly_vec(vec: np.ndarray, modulus: int) -> bytes:
    """
    Pack a polynomial vector shaped (k, n) with coefficients in [0, modulus).
    """
    vec = np.asarray(vec, dtype=params.DTYPE)
    bits = modulus.bit_length() - 1
    return pack_bits_le(vec % modulus, bits)


def pack_poly(poly: np.ndarray, modulus: int) -> bytes:
    poly = np.asarray(poly, dtype=params.DTYPE)
    bits = modulus.bit_length() - 1
    return pack_bits_le(poly % modulus, bits)

