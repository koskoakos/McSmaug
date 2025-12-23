import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import params
import pke
import sampling
import codec


def H(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def G(mu: bytes, pk_hash: bytes) -> Tuple[bytes, bytes]:
    shake = hashlib.shake_256(mu + pk_hash)
    out = shake.digest(params.HASH_BYTES + params.SEED_BYTES)
    return out[: params.HASH_BYTES], out[params.HASH_BYTES :]


@dataclass
class KEMSecretKey:
    sk_pke: pke.PKESecretKey
    d: bytes
    pk: pke.PKEPublicKey


def serialize_pk(pk: pke.PKEPublicKey) -> bytes:
    if "KAT" in params.ACTIVE.name:
        return pk.seedA + _pack_poly_vec_ref(pk.b)
    return pk.seedA + codec.pack_poly_vec(pk.b, modulus=params.Q)


def serialize_ct(ct: Tuple[object, object]) -> bytes:
    c1, c2 = ct
    return codec.pack_poly_vec(c1, modulus=params.P) + codec.pack_poly(c2, modulus=params.P_PRIME)


def serialize_sk(sk: KEMSecretKey) -> bytes:
    # Match reference KAT format: s_idx (per poly) + neg_start + d
    out = bytearray()
    neg_start = bytearray()
    weight_per_poly = params.H_S // params.K
    s = sk.sk_pke.s
    for i in range(params.K):
        res = bytearray(weight_per_poly)
        res_len = weight_per_poly
        pos_count = 0
        for j in range(params.N):
            v = int(s[i, j])
            if v == 1:
                res[pos_count] = j
                pos_count += 1
            elif v == -1:
                res_len -= 1
                res[res_len] = j
        if pos_count + (weight_per_poly - res_len) != weight_per_poly:
            raise ValueError(
                f"Expected {weight_per_poly} nonzeros in poly {i}, got {pos_count + (weight_per_poly - res_len)}"
            )
        out += res
        neg_start.append(pos_count)
    out += bytes(neg_start)
    out += sk.d
    return bytes(out)


def _pack_poly_vec_ref(vec) -> bytes:
    log_q = params.Q.bit_length() - 1
    poly_bytes = (log_q * params.N) // 8
    out = bytearray()
    for i in range(params.K):
        out += _rq_poly_to_bytes(vec[i], poly_bytes)
    return bytes(out)


def _rq_poly_to_bytes(poly, poly_bytes: int) -> bytes:
    out = bytearray(poly_bytes)
    for i in range(params.N // 4):
        d_idx = 4 * i
        b_idx = 5 * i
        d0 = int(poly[d_idx + 0]) & 0x3FF
        d1 = int(poly[d_idx + 1]) & 0x3FF
        d2 = int(poly[d_idx + 2]) & 0x3FF
        d3 = int(poly[d_idx + 3]) & 0x3FF
        out[b_idx + 0] = d0 & 0xFF
        out[b_idx + 1] = d1 & 0xFF
        out[b_idx + 2] = d2 & 0xFF
        out[b_idx + 3] = d3 & 0xFF
        out[b_idx + 4] = ((d0 >> 8) & 0x03) | ((d1 >> 6) & 0x0C) | ((d2 >> 4) & 0x30) | ((d3 >> 2) & 0xC0)
    return bytes(out)


def keygen(seed: Optional[bytes] = None) -> Tuple[pke.PKEPublicKey, KEMSecretKey]:
    pk, sk_pke = pke.keygen(seed)
    d = sampling.random_seed()
    return pk, KEMSecretKey(sk_pke=sk_pke, d=d, pk=pk)


def encaps(pk: pke.PKEPublicKey) -> Tuple[Tuple[object, object], bytes]:
    mu = sampling.random_bytes(params.MU_BYTES)
    pk_hash = H(serialize_pk(pk))
    K, seed = G(mu, pk_hash)
    ct = pke.encrypt_bytes(pk, mu, seedr=seed)
    return ct, K


def decaps(sk: KEMSecretKey, ct: Tuple[object, object]) -> bytes:
    pk_hash = H(serialize_pk(sk.pk))
    mu_prime = pke.decrypt_to_bytes(sk.sk_pke, ct, params.MU_BYTES)
    K_prime, seed_prime = G(mu_prime, pk_hash)
    ct_prime = pke.encrypt_bytes(sk.pk, mu_prime, seedr=seed_prime)
    if serialize_ct(ct) != serialize_ct(ct_prime):  # Fujisaki-Okamoto
        K_prime, _ = G(sk.d, H(serialize_ct(ct)))  # return random key
    return K_prime
