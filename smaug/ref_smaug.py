from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Tuple

import numpy as np

import rng


@dataclass(frozen=True)
class RefParams:
    mode: int
    k: int
    hs: int
    hr: int
    log_q: int = 10
    log_p: int = 8
    log_t: int = 1
    n: int = 256

    @property
    def q(self) -> int:
        return 1 << self.log_q

    @property
    def p(self) -> int:
        return 1 << self.log_p

    @property
    def t(self) -> int:
        return 1 << self.log_t

    @property
    def _16_log_q(self) -> int:
        return 16 - self.log_q

    @property
    def _16_log_p(self) -> int:
        return 16 - self.log_p

    @property
    def _16_log_t(self) -> int:
        return 16 - self.log_t

    @property
    def rd_add(self) -> int:
        return 0x80

    @property
    def rd_and(self) -> int:
        return 0xFF00

    @property
    def dec_add(self) -> int:
        return 0x4000

    @property
    def delta_bytes(self) -> int:
        return self.n // 8

    @property
    def crypto_bytes(self) -> int:
        return 32

    @property
    def pkseed_bytes(self) -> int:
        return 32

    @property
    def pkpoly_bytes(self) -> int:
        return (self.log_q * self.n) // 8  # 320

    @property
    def pkpolyvec_bytes(self) -> int:
        return self.pkpoly_bytes * self.k

    @property
    def publickey_bytes(self) -> int:
        return self.pkseed_bytes + self.pkpolyvec_bytes

    @property
    def ctpoly_bytes(self) -> int:
        return self.n

    @property
    def ctpolyvec_bytes(self) -> int:
        return self.ctpoly_bytes * self.k

    @property
    def ciphertext_bytes(self) -> int:
        return self.ctpolyvec_bytes + self.ctpoly_bytes

    @property
    def skpoly_bytes(self) -> int:
        return self.hs

    @property
    def skpolyvec_bytes(self) -> int:
        return self.skpoly_bytes * self.k

    @property
    def pke_secretkey_bytes(self) -> int:
        return self.skpolyvec_bytes + self.k  # + neg_start vector

    @property
    def kem_secretkey_bytes(self) -> int:
        return self.pke_secretkey_bytes + self.delta_bytes  # + t


PARAMS_BY_MODE = {
    1: RefParams(mode=1, k=2, hs=70, hr=66),
    3: RefParams(mode=3, k=3, hs=50, hr=49),
    5: RefParams(mode=5, k=5, hs=29, hr=28),
}


def shake128(outlen: int, data: bytes) -> bytes:
    return hashlib.shake_128(data).digest(outlen)


def shake256(outlen: int, data: bytes) -> bytes:
    return hashlib.shake_256(data).digest(outlen)


def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def _u64_le(buf8: bytes) -> int:
    return int.from_bytes(buf8, "little")


def hwt(par: RefParams, input_bytes: bytes, hmwt: int) -> np.ndarray:
    """
    Port of reference_implementation/src/hwt.c.
    Returns dense polynomial over {-1,0,1} encoded as uint8: 0x01 and 0xFF.
    """
    res = np.zeros(par.n, dtype=np.uint8)
    hash_t = bytearray(shake256(par.n >> 3, input_bytes))  # 32 bytes
    hw = 0
    hash_idx = 0
    while hw < hmwt:
        word = _u64_le(hash_t[8 * hash_idx : 8 * (hash_idx + 1)])

        def try_set(deg: int, sign_bit: int) -> None:
            nonlocal hw
            if res[deg] == 0:
                res[deg] = ((sign_bit & 0x02) - 1) & 0xFF
                hw += 1

        try_set(word & 0xFF, (word >> 8) & 0xFF)
        if hw == hmwt:
            hash_idx += 1
            break

        try_set((word >> 10) & 0xFF, (word >> 18) & 0xFF)
        if hw == hmwt:
            hash_idx += 1
            break

        try_set((word >> 20) & 0xFF, (word >> 28) & 0xFF)
        if hw == hmwt:
            hash_idx += 1
            break

        try_set((word >> 30) & 0xFF, (word >> 38) & 0xFF)
        if hw == hmwt:
            hash_idx += 1
            break

        try_set((word >> 40) & 0xFF, (word >> 48) & 0xFF)
        if hw == hmwt:
            hash_idx += 1
            break

        try_set((word >> 50) & 0xFF, (word >> 58) & 0xFF)
        hash_idx += 1

        if hash_idx == (par.n // 64):
            hash_idx = 0
            hash_t = bytearray(shake256(par.n >> 3, bytes(hash_t)))

    return res


def conv_to_idx(op: np.ndarray, out_len: int) -> Tuple[np.ndarray, int]:
    """
    Port of convToIdx from reference_implementation/src/poly.c.
    Returns (idx_array, neg_start).
    """
    res = np.zeros(out_len, dtype=np.uint8)
    neg_start = 0
    res_length_temp = out_len
    for i in range(op.shape[0]):
        v = int(op[i])
        if v == 0x01:
            res[neg_start] = i
            neg_start += 1
        elif v == 0xFF:
            res_length_temp -= 1
            res[res_length_temp] = i
    return res, neg_start


def _bytes_to_rq_poly(par: RefParams, buf: bytes) -> np.ndarray:
    """
    Port of bytes_to_Rq for a single polynomial (PKPOLY_BYTES -> 256 coeffs).
    Produces uint16 coefficients in [0, 1024) (not shifted).
    """
    data = np.zeros(par.n, dtype=np.uint16)
    for i in range(par.n // 4):
        b_idx = 5 * i
        d_idx = 4 * i
        b4 = buf[b_idx + 4]
        data[d_idx + 0] = ((b4 & 0x03) << 8) | buf[b_idx + 0]
        data[d_idx + 1] = ((b4 & 0x0C) << 6) | buf[b_idx + 1]
        data[d_idx + 2] = ((b4 & 0x30) << 4) | buf[b_idx + 2]
        data[d_idx + 3] = ((b4 & 0xC0) << 2) | buf[b_idx + 3]
    return data


def _rq_poly_to_bytes(par: RefParams, data: np.ndarray) -> bytes:
    """
    Port of Rq_to_bytes for a single polynomial (256 coeffs -> PKPOLY_BYTES).
    Expects data in [0, 1024).
    """
    data = np.asarray(data, dtype=np.uint16)
    out = bytearray(par.pkpoly_bytes)
    for i in range(par.n // 4):
        b_idx = 5 * i
        d_idx = 4 * i
        d0 = int(data[d_idx + 0])
        d1 = int(data[d_idx + 1])
        d2 = int(data[d_idx + 2])
        d3 = int(data[d_idx + 3])
        out[b_idx + 0] = d0 & 0xFF
        out[b_idx + 1] = d1 & 0xFF
        out[b_idx + 2] = d2 & 0xFF
        out[b_idx + 3] = d3 & 0xFF
        out[b_idx + 4] = ((d0 >> 8) & 0x03) | ((d1 >> 6) & 0x0C) | ((d2 >> 4) & 0x30) | ((d3 >> 2) & 0xC0)
    return bytes(out)


def gen_ax(par: RefParams, seed: bytes) -> np.ndarray:
    """
    Port of genAx from reference_implementation/src/key.c.
    Returns A with shape (k, k, n) in uint16 shifted by _16_log_q.
    """
    if len(seed) != par.pkseed_bytes:
        raise ValueError("seed must be 32 bytes")
    pkpolymat_bytes = par.pkpoly_bytes * par.k * par.k
    buf = shake128(pkpolymat_bytes, seed)
    A = np.zeros((par.k, par.k, par.n), dtype=np.uint16)
    offset = 0
    for i in range(par.k):
        for j in range(par.k):
            poly_bytes = buf[offset : offset + par.pkpoly_bytes]
            offset += par.pkpoly_bytes
            poly = _bytes_to_rq_poly(par, poly_bytes)
            A[i, j, :] = (poly.astype(np.uint16) << par._16_log_q)
    return A


def _add_gaussian_error_vec(par: RefParams, seed32: bytes) -> np.ndarray:
    """
    Port of addGaussianErrorVec/addGaussianError for NOISE_D1.
    Returns e with shape (k, n) in uint16 shifted by _16_log_q.
    """
    # NOISE_D1 => RAND_BITS=10
    rand_bits = 10
    seed_len_u64 = (rand_bits * par.n) // 64  # 40

    out = np.zeros((par.k, par.n), dtype=np.uint16)
    for i in range(par.k):
        nonce = par.k * i
        seed_tmp = seed32 + int(nonce).to_bytes(8, "little")
        seed_stream = shake128(seed_len_u64 * 8, seed_tmp)
        x = np.frombuffer(seed_stream, dtype="<u8")  # 40 u64

        j = 0
        for base in range(0, par.n, 64):
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = (int(x[j + t]) for t in range(10))

            s0 = (
                (x0 & x1 & x2 & x3 & x4 & x5 & x7 & (~x8)) |
                (x0 & x3 & x4 & x5 & x6 & x8) |
                (x1 & x3 & x4 & x5 & x6 & x8) |
                (x2 & x3 & x4 & x5 & x6 & x8) |
                ((~x2) & (~x3) & (~x6) & x8) |
                ((~x1) & (~x3) & (~x6) & x8) |
                (x6 & x7 & (~x8)) |
                ((~x5) & (~x6) & x8) |
                ((~x4) & (~x6) & x8) |
                ((~x7) & x8)
            ) & ((1 << 64) - 1)

            s1 = (
                (x1 & x2 & x4 & x5 & x7 & x8) |
                (x3 & x4 & x5 & x7 & x8) |
                (x6 & x7 & x8)
            ) & ((1 << 64) - 1)

            for kbit in range(64):
                val = ((s0 >> kbit) & 1) | (((s1 >> kbit) & 1) << 1)
                sign = (x9 >> kbit) & 1
                signed = (((-sign) ^ val) + sign) & 0xFFFF
                out[i, base + kbit] = (signed << par._16_log_q) & 0xFFFF

            j += rand_bits
    return out


def _poly_reduce(par: RefParams, res: np.ndarray, temp: np.ndarray) -> None:
    res[: par.n] = (res[: par.n] + temp[: par.n] - temp[par.n : 2 * par.n]).astype(np.uint16)


def _poly_mult_add(par: RefParams, res: np.ndarray, op1: np.ndarray, op2_idx: np.ndarray, neg_start: int) -> None:
    temp = np.zeros(par.n * 2, dtype=np.uint16)
    for j in range(neg_start):
        deg = int(op2_idx[j])
        temp[deg : deg + par.n] = (temp[deg : deg + par.n] + op1).astype(np.uint16)
    for j in range(neg_start, op2_idx.shape[0]):
        deg = int(op2_idx[j])
        temp[deg : deg + par.n] = (temp[deg : deg + par.n] - op1).astype(np.uint16)
    _poly_reduce(par, res, temp)


def _poly_mult_sub(par: RefParams, res: np.ndarray, op1: np.ndarray, op2_idx: np.ndarray, neg_start: int) -> None:
    temp = np.zeros(par.n * 2, dtype=np.uint16)
    for j in range(neg_start):
        deg = int(op2_idx[j])
        temp[deg : deg + par.n] = (temp[deg : deg + par.n] - op1).astype(np.uint16)
    for j in range(neg_start, op2_idx.shape[0]):
        deg = int(op2_idx[j])
        temp[deg : deg + par.n] = (temp[deg : deg + par.n] + op1).astype(np.uint16)
    _poly_reduce(par, res, temp)


def _matrix_vec_mult_add(par: RefParams, res: np.ndarray, mat: np.ndarray, vec_idx: np.ndarray, vec_neg: np.ndarray, transpose: int) -> None:
    for i in range(par.k):
        for j in range(par.k):
            op1 = mat[j, i] if transpose == 1 else mat[i, j]
            _poly_mult_add(par, res[i], op1, vec_idx[j], int(vec_neg[j]))


def _matrix_vec_mult_sub(par: RefParams, res: np.ndarray, mat: np.ndarray, vec_idx: np.ndarray, vec_neg: np.ndarray, transpose: int) -> None:
    for i in range(par.k):
        for j in range(par.k):
            op1 = mat[j, i] if transpose == 1 else mat[i, j]
            _poly_mult_sub(par, res[i], op1, vec_idx[j], int(vec_neg[j]))


def _vec_vec_mult_add(par: RefParams, res: np.ndarray, vec: np.ndarray, idx_vec: np.ndarray, neg_vec: np.ndarray) -> None:
    for j in range(par.k):
        _poly_mult_add(par, res, vec[j], idx_vec[j], int(neg_vec[j]))


def _gen_s_vec(par: RefParams, seed64: bytes) -> Tuple[np.ndarray, np.ndarray]:
    s_idx = np.zeros((par.k, par.hs), dtype=np.uint8)
    neg_start = np.zeros(par.k, dtype=np.uint8)
    # Replicate reference bug: same seed for each component (no per-i nonce)
    xof_res = shake128(par.crypto_bytes, seed64[: par.crypto_bytes + 8])
    dense = hwt(par, xof_res, par.hs)
    idx, neg = conv_to_idx(dense, par.hs)
    for i in range(par.k):
        s_idx[i] = idx
        neg_start[i] = neg
    return s_idx, neg_start


def _gen_b(par: RefParams, A: np.ndarray, s_idx: np.ndarray, s_neg: np.ndarray, err_seed32: bytes) -> np.ndarray:
    b = _add_gaussian_error_vec(par, err_seed32)
    _matrix_vec_mult_sub(par, b, A, s_idx, s_neg, transpose=0)
    return b


def _pack_pk(par: RefParams, seedA: bytes, b: np.ndarray) -> bytes:
    if len(seedA) != par.pkseed_bytes:
        raise ValueError("seedA must be 32 bytes")
    out = bytearray()
    out += seedA
    for i in range(par.k):
        vec = (b[i] >> par._16_log_q).astype(np.uint16)
        out += _rq_poly_to_bytes(par, vec)
    return bytes(out)


def _pack_sk_pke(par: RefParams, s_idx: np.ndarray, s_neg: np.ndarray) -> bytes:
    out = bytearray()
    for i in range(par.k):
        out += bytes(s_idx[i].tolist())
    out += bytes(s_neg.tolist())
    return bytes(out)


def _pack_sk_kem(par: RefParams, sk_pke: bytes, t: bytes) -> bytes:
    return sk_pke + t


def _pack_ct(par: RefParams, c1: np.ndarray, c2: np.ndarray) -> bytes:
    out = bytearray()
    for i in range(par.k):
        out += bytes((c1[i] & 0xFF).astype(np.uint8).tolist())
    out += bytes((c2 & 0xFF).astype(np.uint8).tolist())
    return bytes(out)


def _unpack_pk(par: RefParams, pk: bytes) -> Tuple[bytes, np.ndarray, np.ndarray]:
    seedA = pk[: par.pkseed_bytes]
    b = np.zeros((par.k, par.n), dtype=np.uint16)
    off = par.pkseed_bytes
    for i in range(par.k):
        poly_b = pk[off : off + par.pkpoly_bytes]
        off += par.pkpoly_bytes
        vec = _bytes_to_rq_poly(par, poly_b)
        b[i] = (vec.astype(np.uint16) << par._16_log_q)
    A = gen_ax(par, seedA)
    return seedA, A, b


def _unpack_sk_pke(par: RefParams, sk: bytes) -> Tuple[np.ndarray, np.ndarray]:
    s_idx = np.zeros((par.k, par.hs), dtype=np.uint8)
    off = 0
    for i in range(par.k):
        s_idx[i] = np.frombuffer(sk[off : off + par.hs], dtype=np.uint8)
        off += par.hs
    s_neg = np.frombuffer(sk[off : off + par.k], dtype=np.uint8).copy()
    return s_idx, s_neg


def _unpack_ct(par: RefParams, ct: bytes) -> Tuple[np.ndarray, np.ndarray]:
    c1 = np.zeros((par.k, par.n), dtype=np.uint16)
    off = 0
    for i in range(par.k):
        c1[i] = np.frombuffer(ct[off : off + par.ctpoly_bytes], dtype=np.uint8).astype(np.uint16)
        off += par.ctpoly_bytes
    c2 = np.frombuffer(ct[off : off + par.ctpoly_bytes], dtype=np.uint8).astype(np.uint16)
    return c1, c2


def indcpa_keypair(par: RefParams) -> Tuple[bytes, bytes]:
    seed = bytearray(par.crypto_bytes + par.pkseed_bytes)
    seed[: par.crypto_bytes] = rng.random_bytes(par.crypto_bytes)
    # shake128(seed, 64, seed, 32)
    expanded = shake128(par.crypto_bytes + par.pkseed_bytes, bytes(seed[: par.crypto_bytes]))
    seed[:] = expanded

    # secret (s, neg_start)
    s_idx, s_neg = _gen_s_vec(par, bytes(seed))
    sk_pke = _pack_sk_pke(par, s_idx, s_neg)

    # pk seed: seed + CRYPTO_BYTES then hashed again inside genPubkey
    pk_seed = bytes(seed[par.crypto_bytes :])
    seedA = shake128(par.pkseed_bytes, pk_seed)
    A = gen_ax(par, seedA)

    b = _gen_b(par, A, s_idx, s_neg, bytes(seed[: par.crypto_bytes]))
    pk = _pack_pk(par, seedA, b)
    return pk, sk_pke


def indcpa_enc(par: RefParams, pk: bytes, delta: bytes) -> bytes:
    seedA, A, b = _unpack_pk(par, pk)
    # r = HWT(delta || H(pk) || nonce) per genRx_vec
    rx_input = bytearray(delta + sha3_256(pk) + b"\x00")
    r_idx = np.zeros((par.k, par.hr), dtype=np.uint8)
    r_neg = np.zeros(par.k, dtype=np.uint8)
    for i in range(par.k):
        rx_input[-1] = i
        dense = hwt(par, bytes(rx_input), par.hr)
        idx, neg = conv_to_idx(dense, par.hr)
        r_idx[i] = idx
        r_neg[i] = neg

    c1 = np.zeros((par.k, par.n), dtype=np.uint16)
    _matrix_vec_mult_add(par, c1, A, r_idx, r_neg, transpose=1)
    # rounding q->p
    c1 = (((c1 + par.rd_add) & par.rd_and) >> par._16_log_p).astype(np.uint16)

    c2 = np.zeros(par.n, dtype=np.uint16)
    # message embedding
    for i in range(par.delta_bytes):
        byte = delta[i]
        for j in range(8):
            c2[8 * i + j] = ((byte >> j) & 1) << par._16_log_t
    _vec_vec_mult_add(par, c2, b, r_idx, r_neg)
    c2 = (((c2 + par.rd_add) & par.rd_and) >> par._16_log_p).astype(np.uint16)

    return _pack_ct(par, c1, c2)


def indcpa_dec(par: RefParams, sk_pke: bytes, ct: bytes) -> bytes:
    s_idx, s_neg = _unpack_sk_pke(par, sk_pke)
    c1, c2 = _unpack_ct(par, ct)
    # expand to 16-bit domain
    c1_16 = (c1.astype(np.uint16) << par._16_log_p).astype(np.uint16)
    c2_16 = (c2.astype(np.uint16) << par._16_log_p).astype(np.uint16)

    delta_temp = c2_16.copy()
    _vec_vec_mult_add(par, delta_temp, c1_16, s_idx, s_neg)
    delta_temp = ((delta_temp + par.dec_add) >> par._16_log_t).astype(np.uint16)

    delta = bytearray(par.delta_bytes)
    for i in range(par.delta_bytes):
        v = 0
        for j in range(8):
            v |= (int(delta_temp[8 * i + j]) & 1) << j
        delta[i] = v
    return bytes(delta)


def kdf(par: RefParams, ctxt: bytes, delta: bytes) -> bytes:
    input_bytes = delta + sha3_256(ctxt)
    return shake256(par.crypto_bytes, input_bytes)


def crypto_kem_keypair(par: RefParams) -> Tuple[bytes, bytes]:
    pk, sk_pke = indcpa_keypair(par)
    t = rng.random_bytes(par.delta_bytes)
    sk = _pack_sk_kem(par, sk_pke, t)
    return pk, sk


def crypto_kem_encap(par: RefParams, pk: bytes) -> Tuple[bytes, bytes]:
    delta = rng.random_bytes(par.delta_bytes)
    ct = indcpa_enc(par, pk, delta)
    ss = kdf(par, ct, delta)
    return ct, ss


def crypto_kem_decap(par: RefParams, sk: bytes, pk: bytes, ct: bytes) -> bytes:
    sk_pke = sk[: par.pke_secretkey_bytes]
    t = sk[par.pke_secretkey_bytes : par.kem_secretkey_bytes]
    delta = indcpa_dec(par, sk_pke, ct)
    ct_check = indcpa_enc(par, pk, delta)
    if ct != ct_check:
        return kdf(par, ct, t)
    return kdf(par, ct, delta)

