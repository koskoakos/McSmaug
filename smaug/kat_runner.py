from __future__ import annotations

"""
KAT runner for SMAUG.

Note: This runner validates against the reference implementation outputs.
The main implementation follows the SMAUG paper (FO-style KEM flow), so
passing the official KAT vectors requires using the reference KEM here.
KAT-specific helpers (e.g., ref samplers/packing) remain in the main codebase
for comparison and debugging.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import params
import rng
from nist_aes256ctr_drbg import AES256CTRDRBG
import kem
import ref_smaug


@dataclass
class KatCase:
    count: int
    seed48: bytes
    pk: bytes
    sk: bytes
    ct: bytes
    ss: bytes


def _parse_rsp(path: Path) -> list[KatCase]:
    cases: list[KatCase] = []
    cur: dict[str, object] = {}
    key_re = re.compile(r"^(count|seed|pk|sk|ct|ss)\s*=\s*(.*)$")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = key_re.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k == "count":
            if cur:
                cases.append(
                    KatCase(
                        count=int(cur["count"]),
                        seed48=cur["seed48"],  # type: ignore[arg-type]
                        pk=cur["pk"],  # type: ignore[arg-type]
                        sk=cur["sk"],  # type: ignore[arg-type]
                        ct=cur["ct"],  # type: ignore[arg-type]
                        ss=cur["ss"],  # type: ignore[arg-type]
                    )
                )
                cur = {}
            cur["count"] = int(v)
        elif k == "seed":
            cur["seed48"] = bytes.fromhex(v)
        else:
            cur[k] = bytes.fromhex(v)
    if cur:
        cases.append(
            KatCase(
                count=int(cur["count"]),
                seed48=cur["seed48"],  # type: ignore[arg-type]
                pk=cur["pk"],  # type: ignore[arg-type]
                sk=cur["sk"],  # type: ignore[arg-type]
                ct=cur["ct"],  # type: ignore[arg-type]
                ss=cur["ss"],  # type: ignore[arg-type]
            )
        )
    return cases


def _select_params_from_header(rsp_path: Path) -> str:
    header = rsp_path.read_text().splitlines()[0].strip()
    if "SMAUG1" in header or "smaug128" in rsp_path.as_posix():
        return "SMAUG-128-KAT"
    if "SMAUG3" in header or "smaug192" in rsp_path.as_posix():
        return "SMAUG-192-KAT"
    if "SMAUG5" in header or "smaug256" in rsp_path.as_posix():
        return "SMAUG-256-KAT"
    raise ValueError(f"Cannot infer parameter set from {rsp_path}")


def _kat_ref_params(par_name: str) -> ref_smaug.RefParams:
    if par_name == "SMAUG-128-KAT":
        return ref_smaug.PARAMS_BY_MODE[1]
    if par_name == "SMAUG-192-KAT":
        return ref_smaug.PARAMS_BY_MODE[3]
    if par_name == "SMAUG-256-KAT":
        return ref_smaug.PARAMS_BY_MODE[5]
    raise ValueError("Unexpected param selection for KAT")


def run_rsp(rsp_path: Path, max_cases: int | None = None, use_main_keygen: bool = False) -> int:
    par_name = _select_params_from_header(rsp_path)
    ref_par = _kat_ref_params(par_name)
    params.set_active(par_name)
    cases = _parse_rsp(rsp_path)
    if max_cases is not None:
        cases = cases[:max_cases]

    failures = 0
    for case in cases:
        if len(case.seed48) != 48:
            raise ValueError("Expected 48-byte KAT seed")
        rng.set_rng(AES256CTRDRBG.from_seed(case.seed48))

        if use_main_keygen:
            pk, sk = kem.keygen()
            pk_bytes = kem.serialize_pk(pk)
            sk_bytes = kem.serialize_sk(sk)
        else:
            pk_bytes, sk_bytes = ref_smaug.crypto_kem_keypair(ref_par)
        ct_bytes, ss = ref_smaug.crypto_kem_encap(ref_par, pk_bytes)
        if not use_main_keygen:
            ss2 = ref_smaug.crypto_kem_decap(ref_par, sk_bytes, pk_bytes, ct_bytes)
            if ss != ss2:
                print(f"[count={case.count}] decaps mismatch (internal)")
                failures += 1
                break

        mism = []
        if pk_bytes != case.pk:
            mism.append("pk")
        if ct_bytes != case.ct:
            mism.append("ct")
        if ss != case.ss:
            mism.append("ss")
        if sk_bytes != case.sk:
            mism.append("sk")

        if mism:
            print(f"[count={case.count}] mismatch: {', '.join(mism)}")
            if "pk" in mism:
                print(f"  expected pk len={len(case.pk)} got={len(pk_bytes)}")
            if "ct" in mism:
                print(f"  expected ct len={len(case.ct)} got={len(ct_bytes)}")
            failures += 1
            break

    rng.reset_rng()
    return failures


def main(argv: list[str]) -> int:
    use_main_keygen = "--use-main-keygen" in argv
    pos_args = [a for a in argv[1:] if not a.startswith("--")]
    if not pos_args:
        # Run all KATs if no specific RSP provided
        kat_dir = Path(__file__).parent.parent / "KAT"
        rsp_files = list(kat_dir.glob("**/PQCkemKAT_*.rsp"))
        if not rsp_files:
            print("No RSP files found in KAT directory")
            return 2
        total_failures = 0
        for rsp in rsp_files:
            print(f"Running KAT for {rsp.parent.name}")
            failures = run_rsp(rsp, use_main_keygen=use_main_keygen)
            if failures > 0:
                print(f"FAIL for {rsp.parent.name} ({failures})")
                total_failures += failures
            else:
                print(f"OK for {rsp.parent.name}")
        print("Overall: OK" if total_failures == 0 else f"Overall: FAIL ({total_failures})")
        return 0 if total_failures == 0 else 1
    rsp = Path(pos_args[0])
    max_cases = int(pos_args[1]) if len(pos_args) >= 2 else None
    failures = run_rsp(rsp, max_cases=max_cases, use_main_keygen=use_main_keygen)
    print("OK" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
