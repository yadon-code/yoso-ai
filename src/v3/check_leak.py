"""リークの点検: ある日（CUT）より後のデータを丸ごと消して特徴量を作り直し、CUT までの行が1つも変わらないことを確かめる。
変わる列があれば「未来のデータを使っている」ので、その列の名前を出して失敗する。

    python src/v3/check_leak.py 2026-06-15      # 15分ほど（特徴量を作り直すため）
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import paths  # noqa: F401
from feats import FEATURES
from paths import KEY, TABLES


def main(cut):
    env = {**os.environ, "YOSO_FEATS_UNTIL": cut, "YOSO_FEATS_FILE": f"feats_until_{cut}.parquet"}
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "feats.py")], env=env, check=True)
    a = pd.read_parquet(TABLES / "feats.parquet")
    b = pd.read_parquet(TABLES / f"feats_until_{cut}.parquet")
    a = a[a.date <= cut].sort_values(KEY + ["boat"]).reset_index(drop=True)
    b = b.sort_values(KEY + ["boat"]).reset_index(drop=True)
    assert len(a) == len(b), (len(a), len(b))
    bad = []
    for c in FEATURES:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        diff = ~((np.isclose(x, y, equal_nan=True)) | (np.isnan(x) & np.isnan(y)))
        if diff.any():
            bad.append((c, int(diff.sum())))
    (TABLES / f"feats_until_{cut}.parquet").unlink()
    if bad:
        print("未来のデータで変わった列:", bad)
        sys.exit(1)
    print(f"OK: {cut} までの {len(a)} 行・{len(FEATURES)} 列は、未来のデータを消しても同じ")


if __name__ == "__main__":
    main(sys.argv[1])
