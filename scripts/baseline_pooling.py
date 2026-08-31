"""attention 없이 단순 pooling + 로지스틱 회귀로 비교 기준선을 세운다.

ABMIL은 attention이 균등할 때 정확히 mean pooling이 된다. 즉 mean pooling은
ABMIL의 특수한 경우이므로, ABMIL이 이 기준선을 넘지 못하면 attention을 쓸 근거가
없다. MIL 연구에서 흔히 빠뜨리는 비교라 명시적으로 남겨둔다.

정규화 세기 C는 val로 고른다. test는 건드리지 않는다.

    python scripts/baseline_pooling.py
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "manifest" / "manifest.csv"
FEATURES_DIR = (
    PROJECT_ROOT / "dataset" / "features" / "20x_512px_0px_overlap" / "features_conch_v15"
)
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def load_split(df: pd.DataFrame, split: str) -> tuple:
    sub = df[df["split"] == split]
    mean, mx, labels = [], [], []
    for fname, y in zip(sub["svs_file"], sub["y"]):
        with h5py.File(FEATURES_DIR / f"{Path(fname).stem}.h5", "r") as f:
            arr = f["features"][:]
        mean.append(arr.mean(0))
        mx.append(arr.max(0))
        labels.append(y)
    return np.array(mean), np.array(mx), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "logs" / "baseline_pooling.json")
    args = parser.parse_args()

    df = pd.read_csv(MANIFEST)
    tr_mean, tr_max, tr_y = load_split(df, "train")
    va_mean, va_max, va_y = load_split(df, "val")
    print(f"train {tr_mean.shape}  val {va_mean.shape}\n")

    results = []
    for pool_name, tr_x, va_x in (
        ("mean", tr_mean, va_mean),
        ("max", tr_max, va_max),
        ("mean+max", np.hstack([tr_mean, tr_max]), np.hstack([va_mean, va_max])),
    ):
        scaler = StandardScaler().fit(tr_x)
        tr_s, va_s = scaler.transform(tr_x), scaler.transform(va_x)
        print(f"{'pooling':<10}{'C':>8}{'train_auc':>11}{'val_auc':>9}")
        for c in C_GRID:
            clf = LogisticRegression(max_iter=5000, C=c).fit(tr_s, tr_y)
            tr_auc = roc_auc_score(tr_y, clf.predict_proba(tr_s)[:, 1])
            va_auc = roc_auc_score(va_y, clf.predict_proba(va_s)[:, 1])
            results.append({"pooling": pool_name, "C": c, "train_auc": tr_auc, "val_auc": va_auc})
            print(f"{pool_name:<10}{c:>8.0e}{tr_auc:>11.4f}{va_auc:>9.4f}")
        print()

    best = max(results, key=lambda r: r["val_auc"])
    print(f"val AUC 최고: {best['pooling']} pooling, C={best['C']:.0e} → {best['val_auc']:.4f}")
    print("\nABMIL은 이 값을 넘어야 attention을 쓸 근거가 생긴다.")

    args.out.write_text(json.dumps({"grid": results, "best": best}, indent=2))
    print(f"기록: {args.out}")


if __name__ == "__main__":
    main()
