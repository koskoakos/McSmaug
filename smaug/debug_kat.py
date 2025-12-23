#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "smaug"))

import numpy as np
import params
import rng
import kem
import pke
import ring
import sampling
import ref_smaug
from nist_aes256ctr_drbg import AES256CTRDRBG


@dataclass
class KatCase:
    count: int
    seed48: bytes
    pk: bytes
    sk: bytes
    ct: bytes
    ss: bytes


CASE0_SMAUG1 = KatCase(
    count=0,
    seed48=bytes.fromhex(
        "061550234D158C5EC95595FE04EF7A25767F2E24CC2BC479D09D86DC9ABCFDE7"
        "056A8C266F9EF97ED08541DBD2E1FFA1"
    ),
    pk=bytes.fromhex(
        "1C10551FF29CCBBD2BC660B01267C0FC918441BCBDDC7EBFEF1EF7162DBDC99F"
        "12B235AD88C3A65D8BF2EC08174074BEBDF43800C6AEEEDC0D84FC46C651AE7A"
        "8184E63B6B6545744A4C7B5A05A3536E7DB0C35505A7FD2B3FF2AB2F373D3658"
        "93F6E6DD39ABADDF44E637515D34678EBC774AC16E244B5B3D44924770D5AF62"
        "A923F6F91B87F79F01B4E0CF231DEF3745958DB5CD23CC963D8375A5C2D795311"
        "869468D4FDDEAB69382BB7CF7F0249B6B241DC892705867072DF91FC87EA747B"
        "39141B4DE0A34F4253AA0F9BAB809B06A4E06FD0AFBA79C9C1BA79C9E502BFCF"
        "101C9C332AE5D71AD576D072DB734493ACCC35CC55A42BFE3A3231FC02B36E1A8"
        "3D6485CF982ACEF90724BB6BD5807CFC4BD8B3080FBD4DAFD6DDCD7EA9B41848"
        "EB3E87A716438EC76B55373A48C51CB728E1AA371D2AE02CD968E9659E929DADA"
        "D9B283378F7DF447E1F72FEDF122F3FF9161DCB6695E7409A3A1676BAE3AA40A1"
        "7F67A4997405EDE2949BD828753E63EC7D1C90866BDD1CD65BB2CCF6CAE3A24DF"
        "828E0FE6946F91D1E0AA1046CEF62D20CE3BB42CC6D6485446417503544C92CCC"
        "FE76DE738386689B47B6D94F512DA4C6A3A750015394C85DA42C5A45F2EB9C89B"
        "CE6CA58646E62DEF10C60AB9021A0CF800E80018606EEAB2A23D89CEBA0F43D0"
        "8DDBAEE0DBA58FC7F0E29499FA17E80318BD7C4F8061E5072F02673AB8DB553E0"
        "EB6C1DA109AADF89D9BCE8BD4777537F65CE52A9E6FC70ABD750DD4398BDBAFF9"
        "AB65F42981AF0950F05AA6AB3DB1A7A193CD30CB0635C18F4F7BE6C6769FAD7E"
        "0A434FD17A50445A27C88DB2144E8D4FE4B04EDD77D693E6519D20DF9B077929D"
        "4A447DBA87F31D2F346C68E8F4CCC5138C8EBA2BF397B234EB7DDD8D3C83A0D91"
        "61F7FACF8D6E6DD78B10E0B12841CB65A32C4BF8CE34A51E0A3A6"
    ),
    sk=bytes.fromhex(
        "0310131F2F31363D404B5364656F71737C86878C8F91999DA0ABADC7D0EBF5F9"
        "FDFBF2EEE6E1E0DFD2D1CFC8C5BFBCAFAEA19B95908B75706B5A5854493C2B27"
        "25232219170E0310131F2F31363D404B5364656F71737C86878C8F91999DA0ABAD"
        "C7D0EBF5F9FDFBF2EEE6E1E0DFD2D1CFC8C5BFBCAFAEA19B95908B75706B5A5854"
        "493C2B2725232219170E21218626ED79D451140800E03B59B956F8210E55606740"
        "7D13DC90FA9E8B872BFB8F"
    ),
    ct=bytes.fromhex(
        "DB73D65A0F852EADE47F3656F776B1718D3E9379D070BD5ED771666A9991D8C0"
        "672D24882292D2D896A5BE1F43681FF9E9D8076CBC08139C51AFE7E4826E27A7"
        "81DE7CEA6D5832F20DD461B4F863310A7D5C890CD9A816AB2D09DDCE4C3046B6B"
        "049889EC40C0E9345695ED9F6C80D5683B5D4C067791B48637B3B9DCD7393B240"
        "07897D78DD376976128D61EA77B687A077DF8232A213CCD02ED5946182CC774A8"
        "A09F1D1C256E174FD952F2FD0D1288F4E84A4CC78B64699C6B7DF8788B8EC7BE8"
        "B1CD217CD8BB04A2F1896695FAF2AAC9A7A31626D159DEEEEA9842F5D519EFED6"
        "D2331506EF8BA40845E4AEEF086615062F3A88507A6C6FB96EA38FE7917943ABE"
        "8CEB12D0C93EE478310C937191D673890A4E990578A538EE63BE5CF41E61EEFA6"
        "9A916412E88FC82850F5B4D1F20CBD0D8B2CDFEB06AE6A8D13618523BDAD0CACF"
        "58A39C9D255D329BCDCF790AF7C36C6B92F8659D203092CB6000F36F74917DA45"
        "382CA6A062F4FC2778A92DEE5318D552B98D93E47DC079474E0D5D3CBBE2392FF"
        "CEFFA8168DBF3AD42207332EA4D1855351261BC44D8D523CC72FA875C46BE7F50"
        "4AE4B872395BA77D36F0C62449505E74153ED4D70AF1E6363B861B473DDE02119"
        "B429781A75E71364307B7DA511637232D058E7435A500596D0225C3280F6FE71F"
        "55D07D2E65D1A4937CCE0371F034A6D16BFFBC37E2CB336BE373129AADD1A19BE"
        "956F1337357596E5AF71374E7F0091A5372C254F0E5868303B0C420460CE4FE79"
        "4D5C61562FEAAAA6A56CEF347422B8690E751EF538E649DECACFE52F5F7BCEA57"
        "43055D6B47FC94157285DDBCE9979E28EF11D5A256823DDB88BD1B9937F744A08"
        "EBE9C9B3047C53B021116A4D2F4D01C46BEC826EC1158CD773DD2975D90B9F388"
        "4A037B416E2DA93829EBC04374723C389D2DF3E40C64D0D72ED0D0A7F67FE9A20"
        "8F249B13B4371632196F8FF4BFA2EF65E61BE5A70CF24147BEA5EBB1DC753B69D"
        "11D71FFA4A9328BD1F1BEF87463C5EC3FD628CEB7F03EE23182A7DAB73682B511"
        "2410F1EE7FB16CA880E3495F9C719A24E9417C692F8"
    ),
    ss=bytes.fromhex("E594791665843673E7C7E8FB52148652767E03BE0E904EA51EB304277266CA1F"),
)


def _run_main(case: KatCase) -> dict[str, bytes]:
    rng.set_rng(AES256CTRDRBG.from_seed(case.seed48))
    pk, sk = kem.keygen()
    pk_bytes = kem.serialize_pk(pk)
    sk_bytes = kem.serialize_sk(sk)
    ct, ss = kem.encaps(pk)
    ct_bytes = kem.serialize_ct(ct)
    ss_dec = kem.decaps(sk, ct)
    rng.reset_rng()
    return {
        "pk": pk_bytes,
        "sk": sk_bytes,
        "ct": ct_bytes,
        "ss": ss,
        "ss_dec": ss_dec,
    }


def _run_ref(case: KatCase) -> dict[str, bytes]:
    rng.set_rng(AES256CTRDRBG.from_seed(case.seed48))
    par = ref_smaug.PARAMS_BY_MODE[1]
    pk, sk = ref_smaug.crypto_kem_keypair(par)
    ct, ss = ref_smaug.crypto_kem_encap(par, pk)
    ss_dec = ref_smaug.crypto_kem_decap(par, sk, pk, ct)
    rng.reset_rng()
    return {
        "pk": pk,
        "sk": sk,
        "ct": ct,
        "ss": ss,
        "ss_dec": ss_dec,
    }


def _cmp(label: str, a: bytes, b: bytes) -> None:
    if a == b:
        print(f"OK {label}")
    else:
        idx = _first_diff_bytes(a, b)
        if idx is None:
            print(f"FAIL {label} (len {len(a)} vs {len(b)})")
        else:
            print(
                f"FAIL {label} (len {len(a)} vs {len(b)}), "
                f"first diff at {idx}: {a[idx]:02X} vs {b[idx]:02X}"
            )


def _first_diff_bytes(a: bytes, b: bytes) -> int | None:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return None


def _cmp_array(label: str, a: np.ndarray, b: np.ndarray) -> None:
    if np.array_equal(a, b):
        print(f"OK {label}")
        return
    diff = np.argwhere(a != b)
    if diff.size == 0:
        print(f"FAIL {label}")
        return
    idx = tuple(diff[0].tolist())
    print(f"FAIL {label} at {idx}: {a[idx]} vs {b[idx]}")


def _dense_to_idx_neg(dense: np.ndarray, weight_per_poly: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.zeros((params.K, weight_per_poly), dtype=np.uint8)
    neg_start = np.zeros(params.K, dtype=np.uint8)
    for i in range(params.K):
        res = np.zeros(weight_per_poly, dtype=np.uint8)
        res_len = weight_per_poly
        pos_count = 0
        for j in range(params.N):
            v = int(dense[i, j])
            if v == 1:
                res[pos_count] = j
                pos_count += 1
            elif v == -1:
                res_len -= 1
                res[res_len] = j
        idx[i, :] = res
        neg_start[i] = pos_count
    return idx, neg_start


def _main_keygen_debug(seed48: bytes) -> dict[str, object]:
    rng.set_rng(AES256CTRDRBG.from_seed(seed48))
    seed = sampling.random_seed()
    seedA, seedsk, seede = pke.split_seed(seed)
    A = sampling.expand_matrix(seedA)
    if "KAT" in params.ACTIVE.name:
        s = sampling.sample_hwt_kat(seedsk, params.H_S)
    else:
        s = sampling.sample_hwt(seedsk, params.H_S)
    e = sampling.sample_discrete_gaussian(seede, params.SIGMA)
    As = ring.mat_vec_mul(A, s, q=params.Q)
    b = (e - As) % params.Q
    weight_per_poly = params.H_S // params.K
    s_idx, s_neg = _dense_to_idx_neg(s, weight_per_poly)
    rng.reset_rng()
    return {
        "seed": seed,
        "seedA": seedA,
        "seedsk": seedsk,
        "seede": seede,
        "A": A,
        "s": s,
        "s_idx": s_idx,
        "s_neg": s_neg,
        "e": e,
        "b": b,
    }


def _ref_keygen_debug(seed48: bytes) -> dict[str, object]:
    rng.set_rng(AES256CTRDRBG.from_seed(seed48))
    par = ref_smaug.PARAMS_BY_MODE[1]
    seed = bytearray(par.crypto_bytes + par.pkseed_bytes)
    seed32 = rng.random_bytes(par.crypto_bytes)
    seed[: par.crypto_bytes] = seed32
    expanded = ref_smaug.shake128(par.crypto_bytes + par.pkseed_bytes, bytes(seed[: par.crypto_bytes]))
    seed[:] = expanded
    s_idx, s_neg = ref_smaug._gen_s_vec(par, bytes(seed))
    pk_seed = bytes(seed[par.crypto_bytes :])
    seedA = ref_smaug.shake128(par.pkseed_bytes, pk_seed)
    A = ref_smaug.gen_ax(par, seedA)
    b = ref_smaug._gen_b(par, A, s_idx, s_neg, bytes(seed[: par.crypto_bytes]))
    rng.reset_rng()
    return {
        "seed32": seed32,
        "seed": bytes(seed),
        "seedA": seedA,
        "s_idx": s_idx,
        "s_neg": s_neg,
        "A": A,
        "b": b,
    }


def _compare_keygen(case: KatCase) -> None:
    print("\n== keygen debug ==")
    main_dbg = _main_keygen_debug(case.seed48)
    ref_dbg = _ref_keygen_debug(case.seed48)

    _cmp("seed32", main_dbg["seed"], ref_dbg["seed32"])
    _cmp("seedA", main_dbg["seedA"], ref_dbg["seedA"])
    _cmp_array("s_idx", main_dbg["s_idx"], ref_dbg["s_idx"])
    _cmp_array("s_neg", main_dbg["s_neg"], ref_dbg["s_neg"])

    log_q = params.Q.bit_length() - 1
    shift = 16 - log_q
    A_main_16 = (main_dbg["A"].astype(np.uint16) << shift) & 0xFFFF
    b_main_16 = (main_dbg["b"].astype(np.uint16) << shift) & 0xFFFF
    _cmp_array("A (shifted)", A_main_16, ref_dbg["A"])
    _cmp_array("b (shifted)", b_main_16, ref_dbg["b"])

    print("\n== HWT sampler debug ==")
    paper_s = sampling.sample_hwt(main_dbg["seedsk"], params.H_S)
    paper_idx, paper_neg = _dense_to_idx_neg(paper_s, params.H_S // params.K)
    _cmp_array("paper s_idx vs ref", paper_idx, ref_dbg["s_idx"])
    _cmp_array("paper s_neg vs ref", paper_neg, ref_dbg["s_neg"])


def main() -> int:
    case = CASE0_SMAUG1
    params.set_active("SMAUG-128-KAT")

    print("== main vs ref ==")
    main_out = _run_main(case)
    ref_out = _run_ref(case)
    _cmp("pk", main_out["pk"], ref_out["pk"])

    _cmp("sk", main_out["sk"], ref_out["sk"])
    _cmp("ct", main_out["ct"], ref_out["ct"])
    _cmp("ss", main_out["ss"], ref_out["ss"])
    _cmp("main:ss_dec", main_out["ss"], main_out["ss_dec"])
    _cmp("ref:ss_dec", ref_out["ss"], ref_out["ss_dec"])

    print("\n== main vs KAT ==")
    _cmp("pk", main_out["pk"], case.pk)
    _cmp("sk", main_out["sk"], case.sk)
    _cmp("ct", main_out["ct"], case.ct)
    _cmp("ss", main_out["ss"], case.ss)

    print("\n== ref vs KAT ==")
    _cmp("pk", ref_out["pk"], case.pk)
    _cmp("sk", ref_out["sk"], case.sk)
    _cmp("ct", ref_out["ct"], case.ct)
    _cmp("ss", ref_out["ss"], case.ss)
    _compare_keygen(case)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
