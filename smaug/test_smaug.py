import numpy as np

import kem
import params
import pke


def test_pke_roundtrip():
    params.set_active("TOY-NOISELESS")
    pk, sk = pke.keygen()
    msg = bytes(range(params.MU_BYTES))
    ct = pke.encrypt_bytes(pk, msg)
    dec = pke.decrypt_to_bytes(sk, ct, len(msg))
    assert dec == msg
    print(f"PKE roundtrip test passed. {msg} -> {dec}")


def test_kem_encap_decaps():
    params.set_active("TOY-NOISELESS")
    pk, sk = kem.keygen()
    ct, K = kem.encaps(pk)
    K_prime = kem.decaps(sk, ct)
    assert K == K_prime
    print(f"KEM encaps/decaps test passed. Shared key: {K.hex()}")


if __name__ == "__main__":
    test_pke_roundtrip()
    test_kem_encap_decaps()
    print("All SMAUG toy tests passed.")
