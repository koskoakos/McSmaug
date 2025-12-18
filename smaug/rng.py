from __future__ import annotations

import secrets
from typing import Protocol, Optional


class RandomBytes(Protocol):
    def random_bytes(self, n: int) -> bytes: ...


class _SystemRNG:
    def random_bytes(self, n: int) -> bytes:
        return secrets.token_bytes(n)


_active: RandomBytes = _SystemRNG()


def set_rng(rng: RandomBytes) -> None:
    global _active
    _active = rng


def reset_rng() -> None:
    global _active
    _active = _SystemRNG()


def random_bytes(n: int) -> bytes:
    return _active.random_bytes(n)


def maybe_get_active() -> Optional[RandomBytes]:
    return _active

