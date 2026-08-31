"""manifest.csv의 split 컬럼을 COMPOUND_NAME 단위로 채운다.

manifest.csv에 나열된 순서를 유지한 채, 화합물 묶음을 앞에서부터 train/val/test에
할당한다. 경계에 걸친 화합물은 6:2:2 목표 슬라이드 수에 더 가까워지는 쪽으로 보낸다.

화합물 단위로 나누는 이유는 누수 차단이다. 같은 화합물은 유발하는 병변 양상이
비슷하므로, train과 test에 함께 등장하면 test 성능이 부풀려진다. COMPOUND_NAME은
manifest에서 가장 큰 묶음이라(화합물 ⊃ EXP_ID ⊃ EXP+GROUP ⊃ pair) 이 기준으로
나누면 하위 단위의 누수도 자동으로 함께 차단된다.

    python scripts/make_split.py
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "manifest" / "manifest.csv"

GROUP_KEY = "COMPOUND_NAME"
SPLITS = ("train", "val", "test")
RATIOS = (0.6, 0.2, 0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    units = pd.unique(df[GROUP_KEY])                      # 파일 등장 순서 유지
    sizes = df.groupby(GROUP_KEY, sort=False).size()
    n = len(df)
    boundaries = [n * RATIOS[0], n * (RATIOS[0] + RATIOS[1])]  # 누적 슬라이드 목표

    assignment = {}
    cum = 0
    stage = 0  # 0=train, 1=val, 2=test
    for unit in units:
        size = sizes[unit]
        # 이 화합물을 현재 split에 더 넣으면 경계에서 오히려 멀어지는 시점에 다음 split으로 넘긴다.
        while stage < 2 and abs(cum + size - boundaries[stage]) > abs(cum - boundaries[stage]):
            stage += 1
        assignment[unit] = SPLITS[stage]
        cum += size

    df["split"] = df[GROUP_KEY].map(assignment)
    df.to_csv(args.manifest, index=False)
    print(f"split 기록 완료: {args.manifest}  (기준: {GROUP_KEY})\n")

    # --- 검증 ---
    print(f"{'split':<7}{'슬라이드':>9}{'비율':>8}{'화합물':>8}{'쌍':>6}{'normal':>8}{'abnormal':>10}")
    for name in SPLITS:
        sub = df[df["split"] == name]
        print(
            f"{name:<7}{len(sub):>9}{100 * len(sub) / n:>7.1f}%"
            f"{sub[GROUP_KEY].nunique():>8}{sub.pair_id.nunique():>6}"
            f"{(sub.y == 0).sum():>8}{(sub.y == 1).sum():>10}"
        )

    assert df["split"].isna().sum() == 0, "split이 비어 있는 행이 있다"
    print()
    for key in ("pair_id", "EXP_ID", "COMPOUND_NAME"):
        spanning = df.groupby(key)["split"].nunique()
        n_bad = int((spanning > 1).sum())
        print(f"  여러 split에 걸친 {key:<14}: {n_bad}개")
        assert n_bad == 0, f"{key}가 split을 넘나든다: {spanning[spanning > 1].index.tolist()}"
    print("\n검증 통과")


if __name__ == "__main__":
    main()
