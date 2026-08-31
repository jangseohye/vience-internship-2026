"""metadata.csv의 max_grade를 manifest.csv에 붙인다.

이진 라벨(normal/abnormal)은 minimal과 severe를 똑같이 1로 취급해 정보를 버린다.
등급을 순서형으로 쓰면 슬라이드당 감독 신호가 훨씬 풍부해진다. 지금 최대 병목이
"606개의 이진 라벨"이므로 이게 가장 직접적인 처방이다.

max_grade는 WSI 하나에 여러 병변이 있을 때 그중 최대 등급이다.

    python scripts/add_grade.py
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "manifest" / "manifest.csv"
METADATA = PROJECT_ROOT / "metadata" / "metadata.csv"

# 순서가 의미를 갖는다. normal < minimal < slight < moderate < severe
GRADE_ORDER = ["normal", "minimal", "slight", "moderate", "severe"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    args = parser.parse_args()

    man = pd.read_csv(args.manifest)
    meta = pd.read_csv(args.metadata)

    cols = ["svs_file", "max_grade", "n_findings", "n_findings_spont", "dose_level"]
    merged = man.merge(meta[cols], on="svs_file", how="left", validate="one_to_one")
    if len(merged) != len(man):
        raise ValueError("join 후 행 수가 달라졌다")

    # normal은 metadata에서 max_grade가 비어 있다. 순서의 맨 아래 칸으로 채운다.
    merged["max_grade"] = merged["max_grade"].fillna("normal")
    bad = set(merged["max_grade"]) - set(GRADE_ORDER)
    if bad:
        raise ValueError(f"알 수 없는 등급: {bad}")

    merged["grade_ord"] = merged["max_grade"].map(GRADE_ORDER.index)

    # y와 등급이 어긋나면 둘 중 하나가 틀린 것이다. 학습 전에 잡아야 한다.
    mismatch = merged[(merged["y"] == 0) != (merged["max_grade"] == "normal")]
    if len(mismatch):
        raise ValueError(f"y와 max_grade가 불일치하는 행 {len(mismatch)}개: "
                         f"{mismatch.svs_file.tolist()[:5]}")

    merged.to_csv(args.manifest, index=False)
    print(f"max_grade, grade_ord, n_findings, dose_level 추가: {args.manifest}\n")

    print(f"{'등급':<10}{'전체':>6}{'train':>7}{'val':>6}{'test':>6}")
    for g in GRADE_ORDER:
        sub = merged[merged["max_grade"] == g]
        print(f"{g:<10}{len(sub):>6}"
              + "".join(f"{int((sub['split'] == s).sum()):>7}" for s in ("train", "val", "test")))


if __name__ == "__main__":
    main()
