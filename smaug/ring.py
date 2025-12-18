import numpy as np

import params


def mod_q(poly, q=params.Q):
    """Reduce coefficients to [0, q)."""
    return np.asarray(poly, dtype=params.DTYPE) % q


def poly_add(a, b, q=params.Q):
    a = np.asarray(a, dtype=params.DTYPE)
    b = np.asarray(b, dtype=params.DTYPE)
    if a.shape != (params.N,) or b.shape != (params.N,):
        raise ValueError(f"Polynomials must have length N={params.N}")
    return (a + b) % q


def poly_sub(a, b, q=params.Q):
    a = np.asarray(a, dtype=params.DTYPE)
    b = np.asarray(b, dtype=params.DTYPE)
    if a.shape != (params.N,) or b.shape != (params.N,):
        raise ValueError(f"Polynomials must have length N={params.N}")
    return (a - b + q) % q


def poly_mul(a, b, q=params.Q):
    """
    Negacyclic multiplication in Z_q[x]/(x^N + 1).
    """
    a = np.asarray(a, dtype=params.DTYPE)
    b = np.asarray(b, dtype=params.DTYPE)
    if a.shape != (params.N,) or b.shape != (params.N,):
        raise ValueError(f"Polynomials must have length N={params.N}")

    t = np.convolve(a, b)
    c = t[: params.N].copy()
    c[: params.N - 1] -= t[params.N :]
    return c % q


def compress(poly, q: int, p: int):
    """
    Round-and-scale from Z_q to Z_p:
      round(p/q * poly)  (mod p)
    Assumes p | q.
    """
    poly = np.asarray(poly, dtype=params.DTYPE) % q
    return ((poly * p + q // 2) // q) % p


def poly_round_scale(poly, num, den, modulus):
    """
    Compute round(num/den * poly) mod modulus.
    """
    poly = np.asarray(poly, dtype=np.float64)
    scaled = np.rint((num / den) * poly)
    return scaled.astype(params.DTYPE) % modulus


def mat_vec_mul(mat, vec, q=params.Q):
    """
    mat: shape (k, k, N); vec: shape (k, N)
    Returns shape (k, N)
    """
    if mat.shape != (params.K, params.K, params.N):
        raise ValueError(f"Matrix must be shape ({params.K}, {params.K}, {params.N})")
    if vec.shape != (params.K, params.N):
        raise ValueError(f"Vector must be shape ({params.K}, {params.N})")
    out = np.zeros((params.K, params.N), dtype=params.DTYPE)
    for i in range(params.K):
        acc = np.zeros(params.N, dtype=params.DTYPE)
        for j in range(params.K):
            acc = poly_add(acc, poly_mul(mat[i, j], vec[j], q=q), q=q)
        out[i] = acc
    return out


def vec_dot(a, b, q=params.Q):
    """
    Inner product of two polynomial vectors of length k in the ring.
    """
    if a.shape != (params.K, params.N) or b.shape != (params.K, params.N):
        raise ValueError(f"Vectors must be shape ({params.K}, {params.N})")
    acc = np.zeros(params.N, dtype=params.DTYPE)
    for i in range(params.K):
        acc = poly_add(acc, poly_mul(a[i], b[i], q=q), q=q)
    return acc
