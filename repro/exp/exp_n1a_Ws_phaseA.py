# -*- coding: utf-8 -*-
"""Phase A: N1-A W_s sensitivity — estimator-only shell around _chk_shat_t0.

Does NOT modify src/corr_est.py. Sweeps SOnline window W via P3_SWIN.

Env:
  P3_SWIN   comma list, default 10,20,30,50,100,200
  P3_MC0    default 30000
  P3_NMC    default 200  (seeds 30000–30199)
  P3_S      static shares, default 0,0.25,0.5,0.75,1
  P3_ARMS   default rgpo_coh_common (estimator is arm-agnostic for angles; keep 1)
  P3_DYNAMIC 1 to also run dynamic 0.5↔1 schedule (default 1)
  P3_OUT    output dir (default exp/out_tlb/n1a_Ws_phaseA)

Usage:
  D:/anaconda/python.exe exp/exp_n1a_Ws_phaseA.py
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp.exp_tlb_online_s import make_case  # noqa: E402
from src.corr_est import est_s_frame, SOnline  # noqa: E402

SWIN = [int(x) for x in os.environ.get("P3_SWIN", "10,20,30,50,100,200").split(",") if x.strip()]
MC0 = int(os.environ.get("P3_MC0", "30000"))
NMC = int(os.environ.get("P3_NMC", "200"))
S_LIST = [float(x) for x in os.environ.get("P3_S", "0,0.25,0.5,0.75,1").split(",") if x.strip()]
ARMS = os.environ.get("P3_ARMS", "rgpo_coh_common").split(",")
DO_DYN = os.environ.get("P3_DYNAMIC", "1") not in ("0", "false", "False")
OUT = Path(os.environ.get("P3_OUT", str(HERE / "out_tlb" / "n1a_Ws_phaseA")))
OUT.mkdir(parents=True, exist_ok=True)

ARM_IDX = {"rgpo_coh_common": 2, "rgpo_coh_diff": 3, "rgpo_incoh": 4}
K_END = 900
T0_K = 500  # t0=10s @50Hz; used only for dynamic schedule breakpoints


def dynamic_s_true(k: int) -> float:
    """0.5 before t0, 1.0 after (protocol A dynamic 0.5↔1)."""
    return 0.5 if k < T0_K else 1.0


def run_static(cd, sig_az, W: int):
    so = SOnline(sig_az, W=W)
    hats = [so.update(cd, k) for k in range(1, K_END + 1)]
    hats = np.asarray(hats, float)
    raw = np.asarray([est_s_frame(cd, k, sig_az) for k in range(1, K_END + 1)], float)
    return dict(
        mean_hat=float(hats.mean()),
        std_hat=float(hats.std()),
        mean_raw=float(raw.mean()),
        std_raw=float(raw.std()),
        mean_hat_pre=float(hats[: T0_K - 1].mean()),
        mean_hat_post=float(hats[T0_K - 1 :].mean()),
    )


def build_dynamic_cand(mc, arm_idx):
    """Rebuild candidates with piecewise share (best-effort): two halves stitched.
    make_case uses constant share; for estimator-only we approximate by
    concatenating frames from s=0.5 and s=1.0 scenes (same mc/arm).
    """
    cd05, sc, kw, arm = make_case(mc, arm_idx, 0.5)
    cd10, _, _, _ = make_case(mc, arm_idx, 1.0)
    cd = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in cd05.items()}
    for key in ("cand_range", "cand_alpha", "cand_beta", "cand_doppler"):
        a = np.asarray(cd05[key])
        b = np.asarray(cd10[key])
        out = a.copy()
        out[T0_K:] = b[T0_K:]
        cd[key] = out
    return cd, sc, kw, arm


def main():
    t0 = time.perf_counter()
    rows = []
    for arm_name in ARMS:
        idx = ARM_IDX[arm_name]
        for mc in range(MC0, MC0 + NMC):
            for s_true in S_LIST:
                cd, sc, kw, arm = make_case(mc, idx, s_true)
                for W in SWIN:
                    st = run_static(cd, kw["sig_az"], W)
                    rows.append(
                        dict(
                            mc=mc,
                            arm=arm_name,
                            profile="static",
                            s_true=s_true,
                            W=W,
                            **st,
                        )
                    )
            if DO_DYN:
                cd, sc, kw, arm = build_dynamic_cand(mc, idx)
                for W in SWIN:
                    st = run_static(cd, kw["sig_az"], W)
                    rows.append(
                        dict(
                            mc=mc,
                            arm=arm_name,
                            profile="dynamic_0.5_to_1",
                            s_true=np.nan,
                            W=W,
                            **st,
                        )
                    )
    df = pd.DataFrame(rows)
    csv_path = OUT / f"phaseA_{MC0}-{MC0+NMC-1}.csv"
    df.to_csv(csv_path, index=False)
    summary = (
        df.groupby(["profile", "s_true", "W"], dropna=False)[["mean_hat", "std_hat", "mean_hat_pre", "mean_hat_post"]]
        .agg(["mean", "median"])
        .reset_index()
    )
    # flatten cols
    summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns.values]
    sum_path = OUT / f"phaseA_summary_{MC0}-{MC0+NMC-1}.csv"
    summary.to_csv(sum_path, index=False)
    meta = dict(
        entry="exp/exp_n1a_Ws_phaseA.py",
        wraps="exp/_chk_shat_t0.py + src.corr_est.SOnline",
        corr_est_modified=False,
        P3_SWIN=SWIN,
        MC0=MC0,
        NMC=NMC,
        S_LIST=S_LIST,
        ARMS=ARMS,
        dynamic=DO_DYN,
        wall_s=round(time.perf_counter() - t0, 2),
        csv=str(csv_path),
        summary=str(sum_path),
        n_rows=len(df),
    )
    (OUT / "META_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
