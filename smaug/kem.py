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
    return pk.seedA + codec.pack_poly_vec(pk.b, modulus=params.Q)


def serialize_ct(ct: Tuple[object, object]) -> bytes:
    c1, c2 = ct
    return codec.pack_poly_vec(c1, modulus=params.P) + codec.pack_poly(c2, modulus=params.P_PRIME)


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
