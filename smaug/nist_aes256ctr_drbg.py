from __future__ import annotations

from dataclasses import dataclass


_SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]

_RCON = [
    0x00,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36,
]


def _xtime(a: int) -> int:
    a <<= 1
    if a & 0x100:
        a ^= 0x11B
    return a & 0xFF


def _mix_single_column(col: list[int]) -> list[int]:
    t = col[0] ^ col[1] ^ col[2] ^ col[3]
    u = col[0]
    col0 = col[0] ^ t ^ _xtime(col[0] ^ col[1])
    col1 = col[1] ^ t ^ _xtime(col[1] ^ col[2])
    col2 = col[2] ^ t ^ _xtime(col[2] ^ col[3])
    col3 = col[3] ^ t ^ _xtime(col[3] ^ u)
    return [col0 & 0xFF, col1 & 0xFF, col2 & 0xFF, col3 & 0xFF]


def _sub_word(w: list[int]) -> list[int]:
    return [_SBOX[b] for b in w]


def _rot_word(w: list[int]) -> list[int]:
    return [w[1], w[2], w[3], w[0]]


def _expand_key_256(key: bytes) -> list[list[int]]:
    if len(key) != 32:
        raise ValueError("AES-256 key must be 32 bytes")
    Nk = 8
    Nb = 4
    Nr = 14

    w: list[list[int]] = []
    for i in range(Nk):
        w.append([key[4 * i + 0], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]])

    for i in range(Nk, Nb * (Nr + 1)):
        temp = w[i - 1].copy()
        if i % Nk == 0:
            temp = _sub_word(_rot_word(temp))
            temp[0] ^= _RCON[i // Nk]
        elif i % Nk == 4:
            temp = _sub_word(temp)
        w.append([w[i - Nk][j] ^ temp[j] for j in range(4)])
    return w


def _add_round_key(state: list[int], w: list[list[int]], round_idx: int) -> None:
    for c in range(4):
        word = w[round_idx * 4 + c]
        for r in range(4):
            state[r + 4 * c] ^= word[r]


def _sub_bytes(state: list[int]) -> None:
    for i in range(16):
        state[i] = _SBOX[state[i]]


def _shift_rows(state: list[int]) -> None:
    # state is column-major: state[r + 4*c]
    rows = [[state[r + 4 * c] for c in range(4)] for r in range(4)]
    for r in range(4):
        rows[r] = rows[r][r:] + rows[r][:r]
    for r in range(4):
        for c in range(4):
            state[r + 4 * c] = rows[r][c]


def _mix_columns(state: list[int]) -> None:
    for c in range(4):
        col = [state[r + 4 * c] for r in range(4)]
        mixed = _mix_single_column(col)
        for r in range(4):
            state[r + 4 * c] = mixed[r]


def aes256_encrypt_block(key: bytes, block16: bytes) -> bytes:
    if len(block16) != 16:
        raise ValueError("AES block must be 16 bytes")
    w = _expand_key_256(key)
    Nr = 14
    state = list(block16)
    _add_round_key(state, w, 0)
    for r in range(1, Nr):
        _sub_bytes(state)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, w, r)
    _sub_bytes(state)
    _shift_rows(state)
    _add_round_key(state, w, Nr)
    return bytes(state)


def _inc_128_be(v: bytes) -> bytes:
    x = int.from_bytes(v, "big")
    x = (x + 1) & ((1 << 128) - 1)
    return x.to_bytes(16, "big")


@dataclass
class AES256CTRDRBG:
    key: bytes
    v: bytes
    reseed_counter: int = 1

    @classmethod
    def from_seed(cls, seed48: bytes) -> "AES256CTRDRBG":
        if len(seed48) != 48:
            raise ValueError("DRBG seed must be 48 bytes")
        key = bytes(32)
        v = bytes(16)
        drbg = cls(key=key, v=v, reseed_counter=1)
        drbg._update(seed48)
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

