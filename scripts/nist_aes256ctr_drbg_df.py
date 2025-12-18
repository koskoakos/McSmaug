from __future__ import annotations

from dataclasses import dataclass

from nist_aes256ctr_drbg import aes256_encrypt_block, _inc_128_be


def _bcc(key: bytes, data: bytes) -> bytes:
    if len(data) % 16 != 0:
        raise ValueError("BCC data must be multiple of 16 bytes")
    chaining = bytes(16)
    for i in range(0, len(data), 16):
        block = data[i : i + 16]
        xored = bytes(a ^ b for a, b in zip(chaining, block))
        chaining = aes256_encrypt_block(key, xored)
    return chaining


def _int32_be(x: int) -> bytes:
    return int(x).to_bytes(4, "big")


def _df(input_string: bytes, no_of_bytes_to_return: int) -> bytes:
    """
    AES-256 CTR_DRBG derivation function (approximation of NIST SP 800-90A df).

    This is used by some NIST KAT harnesses; it's included here to support
    auto-detection against provided KAT vectors.
    """
    if no_of_bytes_to_return <= 0:
        raise ValueError("no_of_bytes_to_return must be > 0")

    # Build S = L || N || input || 0x80 || 0x00... (pad to 16-byte multiple)
    L = _int32_be(len(input_string))
    N = _int32_be(no_of_bytes_to_return)
    S = L + N + input_string + b"\x80"
    if len(S) % 16 != 0:
        S += b"\x00" * (16 - (len(S) % 16))

    key = bytes(32)  # K = 0^256
    # temp length must cover key (32) + X (16)
    temp_len = 48
    temp = bytearray()

    i = 0
    while len(temp) < temp_len:
        i += 1
        iv = bytes(12) + _int32_be(i)
        temp.extend(_bcc(key, iv + S))

    K = bytes(temp[:32])
    X = bytes(temp[32:48])

    out = bytearray()
    while len(out) < no_of_bytes_to_return:
        X = aes256_encrypt_block(K, X)
        out.extend(X)
    return bytes(out[:no_of_bytes_to_return])


@dataclass
class AES256CTRDRBG_DF:
    key: bytes
    v: bytes
    reseed_counter: int = 1

    @classmethod
    def from_seed(cls, seed48: bytes) -> "AES256CTRDRBG_DF":
        if len(seed48) != 48:
            raise ValueError("DRBG seed must be 48 bytes")
        # Apply derivation function to get 48-byte seed material.
        seed_material = _df(seed48, 48)
        key = bytes(32)
        v = bytes(16)
        drbg = cls(key=key, v=v, reseed_counter=1)
        drbg._update(seed_material)
        return drbg

    def _update(self, provided_data: bytes | None) -> None:
        temp = bytearray()
        v = self.v
        for _ in range(3):
            v = _inc_128_be(v)
            temp.extend(aes256_encrypt_block(self.key, v))
        if provided_data is not None:
            if len(provided_data) != 48:
                raise ValueError("provided_data must be 48 bytes")
            for i in range(48):
                temp[i] ^= provided_data[i]
        self.key = bytes(temp[:32])
        self.v = bytes(temp[32:48])

    def random_bytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            self.v = _inc_128_be(self.v)
            out.extend(aes256_encrypt_block(self.key, self.v))
        self._update(None)
        self.reseed_counter += 1
        return bytes(out[:n])

