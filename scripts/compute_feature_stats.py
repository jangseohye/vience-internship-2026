"""train split의 패치 특징에 대해 차원별 평균과 표준편차를 계산해 저장한다.

CONCH v1.5 출력은 패치 하나 안에서 768차원에 걸쳐 LayerNorm된 상태다. 하지만
데이터셋 전체를 놓고 보면 차원마다 평균이 0이 아니고 분산도 제각각이다. 실제로
평균 pooling 벡터에서 차원별 평균의 절대값이 표준편차의 3.7배였다. 즉 슬라이드를
구분하는 신호가 훨씬 큰 공통 성분에 묻혀 있어 경사하강법의 조건이 나쁘다.

여기서 구한 통계로 표준화하면 sklearn의 StandardScaler와 같은 효과를 낸다.
통계는 반드시 train에서만 계산한다. val/test가 섞이면 누수다.

    python scripts/compute_feature_stats.py
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "manifest" / "manifest.csv"
FEATURES_DIR = (
    PROJECT_ROOT / "dataset" / "features" / "20x_512px_0px_overlap" / "features_conch_v15"
)
DEFAULT_OUT = FEATURES_DIR.parent / "feature_stats_train.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(MANIFEST)
    train = df[df["split"] == "train"]

    # 230만 패치를 한 번에 올리지 않도록 누적합으로 계산한다.
    total = np.zeros(768, dtype=np.float64)
    total_sq = np.zeros(768, dtype=np.float64)
    n = 0
    for fname in train["svs_file"]:
        with h5py.File(FEATURES_DIR / f"{Path(fname).stem}.h5", "r") as f:
            arr = f["features"][:].astype(np.float64)
        total += arr.sum(0)
        total_sq += (arr ** 2).sum(0)
        n += arr.shape[0]

    mean = total / n
    var = total_sq / n - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    print(f"train 슬라이드 {len(train)}장, 패치 {n:,}개")
    print(f"  차원별 평균  절대값 평균 {np.abs(mean).mean():.4f}  범위 [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  차원별 표준편차 평균     {std.mean():.4f}  범위 [{std.min():.3f}, {std.max():.3f}]")

    # 표준편차가 0에 가까운 차원은 나눗셈에서 폭발하므로 하한을 둔다.
    floor = 1e-6
    n_small = int((std < floor).sum())
    if n_small:
        print(f"  표준편차가 {floor} 미만인 차원 {n_small}개 → 1.0으로 대체")
    std = np.where(std < floor, 1.0, std)

    np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32), n_patches=n)
    print(f"\n기록: {args.out}")


if __name__ == "__main__":
    main()
