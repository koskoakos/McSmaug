import numpy as np
import pytest

import kem
import params
import pke
import rng
import sampling
import codec
from nist_aes256ctr_drbg import AES256CTRDRBG
import ring


@pytest.mark.pke
def test_pke_roundtrip():
    params.set_active("TOY-NOISELESS")
    pk, sk = pke.keygen()
    msg = bytes(range(params.MU_BYTES))
    ct = pke.encrypt_bytes(pk, msg)
    dec = pke.decrypt_to_bytes(sk, ct, len(msg))
    assert dec == msg


@pytest.mark.pke
def test_pke_wrong_key():
    params.set_active("TOY-NOISELESS")
    pk1, sk1 = pke.keygen()
    pk2, sk2 = pke.keygen()
    msg = bytes(range(params.MU_BYTES))
    ct = pke.encrypt_bytes(pk1, msg)
    dec_wrong = pke.decrypt_to_bytes(sk2, ct, len(msg))
    assert dec_wrong != msg
    dec_right = pke.decrypt_to_bytes(sk1, ct, len(msg))
    assert dec_right == msg


@pytest.mark.pke
def test_pke_ciphertext_tamper():
    params.set_active("TOY-NOISELESS")
    pk, sk = pke.keygen()
    msg = bytes(range(params.MU_BYTES))
    c1, c2 = pke.encrypt_bytes(pk, msg)
    c1 = c1.copy()
    c1[0, 0] = (int(c1[0, 0]) + 1) % params.P
    ct_bad = (c1, c2)
    dec_bad = pke.decrypt_to_bytes(sk, ct_bad, len(msg))
    assert dec_bad != msg
@pytest.mark.kem
def test_kem_encap_decaps():
    params.set_active("TOY-NOISELESS")
    pk, sk = kem.keygen()
    ct, K = kem.encaps(pk)
    K_prime = kem.decaps(sk, ct)
    assert K == K_prime


def _centered_mod_q(values: np.ndarray, q: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.int64) % q
    half = q // 2
    return np.where(vals >= half, vals - q, vals)


@pytest.mark.sampler
def test_hwt_sampler_stats():
    params.set_active("SMAUG-128")
    seed = bytes(range(32))
    s = sampling.sample_hwt(seed, params.H_S)
    weight_per_poly = params.H_S // params.K
    for i in range(params.K):
        nonzeros = np.count_nonzero(s[i])
        assert nonzeros == weight_per_poly
        assert np.all(np.isin(s[i], [-1, 0, 1]))
    # Ensure per-poly seeds actually differ in non-KAT mode
    assert not np.array_equal(s[0], s[1])


@pytest.mark.sampler
def test_gaussian_sampler_stats():
    params.set_active("SMAUG-128")
    rng.set_rng(AES256CTRDRBG.from_seed(bytes(range(48))))
    samples = []
    for _ in range(32):
        seed = sampling.random_seed()
        e = sampling.sample_discrete_gaussian(seed, params.SIGMA)
        samples.append(_centered_mod_q(e, params.Q))
    rng.reset_rng()
    vals = np.concatenate(samples, axis=None).astype(np.int64)
    mean = float(np.mean(vals))
    var = float(np.var(vals))
    # Deterministic DGS should be symmetric with small variance near sigma^2.
    assert abs(mean) < 0.2
    assert 0.5 < var < 2.5


@pytest.mark.codec
def test_codec_roundtrip_and_edges():
    # Roundtrip for multiple bit widths and edge values
    for bits in (1, 2, 3, 5, 8, 10):
        max_val = (1 << bits) - 1
        values = np.array([0, 1, max_val, max_val - 1, 2, 3], dtype=np.int64)
        packed = codec.pack_bits_le(values, bits)
        unpacked = codec.unpack_bits_le(packed, len(values), bits)
        assert np.array_equal(values & max_val, unpacked)

    # Non-byte-aligned lengths (e.g., 10-bit values)
    values = np.arange(0, 19, dtype=np.int64) & ((1 << 10) - 1)
    packed = codec.pack_bits_le(values, 10)
    unpacked = codec.unpack_bits_le(packed, len(values), 10)
    assert np.array_equal(values, unpacked)

    # Invalid formats: insufficient bytes
    bad_buf = bytes([0xFF])
    try:
        codec.unpack_bits_le(bad_buf, 4, 5)
        assert False, "Expected ValueError for insufficient bytes"
    except ValueError:
        pass


@pytest.mark.kem
def test_kem_wrong_key():
    params.set_active("TOY-NOISELESS")
    pk1, sk1 = kem.keygen()
    pk2, sk2 = kem.keygen()
    ct, K = kem.encaps(pk1)
    K_wrong = kem.decaps(sk2, ct)
    assert K_wrong != K
    # Sanity: correct key still works
    assert kem.decaps(sk1, ct) == K


@pytest.mark.kem
def test_kem_ciphertext_tamper():
    params.set_active("TOY-NOISELESS")
    pk, sk = kem.keygen()
    ct, K = kem.encaps(pk)
    c1, c2 = ct
    c1 = c1.copy()
    c1[0, 0] = (int(c1[0, 0]) + 1) % params.P
    ct_bad = (c1, c2)
    K_bad = kem.decaps(sk, ct_bad)
    assert K_bad != K


@pytest.mark.kem
def test_kem_replay_same_ct():
    params.set_active("TOY-NOISELESS")
    pk, sk = kem.keygen()
    ct, K = kem.encaps(pk)
    assert kem.decaps(sk, ct) == K
    assert kem.decaps(sk, ct) == K


@pytest.mark.kem
def test_kem_random_ct_rejection():
    params.set_active("TOY-NOISELESS")
    pk, sk = kem.keygen()
    ct, K = kem.encaps(pk)
    c1, c2 = ct
    rng.set_rng(AES256CTRDRBG.from_seed(bytes(range(48))))
    c1_rand = rng.random_bytes(params.K * params.N)
    c2_rand = rng.random_bytes(params.N)
    rng.reset_rng()
    c1_rand = np.frombuffer(c1_rand, dtype=np.uint8).astype(np.int64).reshape((params.K, params.N))
    c2_rand = np.frombuffer(c2_rand, dtype=np.uint8).astype(np.int64)
    ct_rand = (c1_rand, c2_rand)
    K_rand = kem.decaps(sk, ct_rand)
    assert K_rand != K


@pytest.mark.determinism
def test_determinism_keygen():
    params.set_active("SMAUG-128")
    seed = bytes(range(32))
    pk1, sk1 = pke.keygen(seed)
    pk2, sk2 = pke.keygen(seed)
    assert np.array_equal(pk1.b, pk2.b)
    assert pk1.seedA == pk2.seedA
    assert np.array_equal(sk1.s, sk2.s)


@pytest.mark.determinism
def test_determinism_encap():
    params.set_active("SMAUG-128")
    rng.set_rng(AES256CTRDRBG.from_seed(bytes(range(48))))
    pk, sk = kem.keygen()
    ct1, K1 = kem.encaps(pk)
    rng.set_rng(AES256CTRDRBG.from_seed(bytes(range(48))))
    pk2, sk2 = kem.keygen()
    ct2, K2 = kem.encaps(pk2)
    rng.reset_rng()
    assert kem.serialize_pk(pk) == kem.serialize_pk(pk2)
    assert kem.serialize_ct(ct1) == kem.serialize_ct(ct2)
    assert K1 == K2


@pytest.mark.ring
def test_ring_add_sub_roundtrip():
    params.set_active("SMAUG-128")
    rng.set_rng(AES256CTRDRBG.from_seed(bytes(range(48))))
    a = np.frombuffer(rng.random_bytes(params.N), dtype=np.uint8).astype(np.int64)
    b = np.frombuffer(rng.random_bytes(params.N), dtype=np.uint8).astype(np.int64)
    rng.reset_rng()
    q = params.Q
    c = ring.poly_add(a, b, q=q)
    back = ring.poly_sub(c, b, q=q)
    assert np.array_equal(back % q, a % q)


@pytest.mark.ring
def test_ring_mul_identity():
    params.set_active("SMAUG-128")
    rng.set_rng(AES256CTRDRBG.from_seed(bytes(range(48))))
    a = np.frombuffer(rng.random_bytes(params.N), dtype=np.uint8).astype(np.int64)
    rng.reset_rng()
    one = np.zeros(params.N, dtype=np.int64)
    one[0] = 1
    prod = ring.poly_mul(a, one, q=params.Q)
    assert np.array_equal(prod % params.Q, a % params.Q)


@pytest.mark.ring
def test_ring_negacyclic_wrap():
    params.set_active("SMAUG-128")
    q = params.Q
    a = np.zeros(params.N, dtype=np.int64)
    b = np.zeros(params.N, dtype=np.int64)
    a[1] = 1
    b[params.N - 1] = 1
    prod = ring.poly_mul(a, b, q=q)
    expected = np.zeros(params.N, dtype=np.int64)
    expected[0] = (q - 1) % q
    assert np.array_equal(prod % q, expected)


@pytest.mark.ring
def test_ring_mod_reduction():
    params.set_active("SMAUG-128")
    q = params.Q
    a = np.array([q, q + 1, -1, -q - 2] + [0] * (params.N - 4), dtype=np.int64)
    reduced = ring.mod_q(a, q=q)
    assert reduced[0] == 0
    assert reduced[1] == 1
    assert reduced[2] == q - 1
    assert reduced[3] == q - 2
