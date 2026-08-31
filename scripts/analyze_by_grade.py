"""학습된 모델을 중증도 등급별로 분해해서 본다.

두 가지를 잰다.
  (1) 성능: 정상 vs 각 등급의 AUC. 전체 AUC 하나로는 "어떤 병변을 못 잡는지" 알 수 없다.
  (2) attention 집중도: 등급이 올라갈수록 중요 패치에 집중하는지.

MIL의 통상적 기대는 "국소 병변일수록 attention이 집중한다"이다. 하지만 TG-GATEs의
등급은 병변 강도만이 아니라 침범 범위를 반영하므로, 등급이 높을수록 이상 패치가
많아져 오히려 분산될 수 있다. 어느 쪽인지는 재봐야 안다.

    python scripts/analyze_by_grade.py logs/ord_lr5e-4/best.pt
    python scripts/analyze_by_grade.py logs/std_lr5e-4/best.pt --splits val test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import WSIBagDataset, MANIFEST
from model import MODELS

GRADES = ["normal", "minimal", "slight", "moderate", "severe"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   help="학습에 쓰이지 않은 split만 넣어야 편향이 없다")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def bootstrap_auc(labels, scores, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[i])) > 1:
            vals.append(roc_auc_score(labels[i], scores[i]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trained = ckpt["args"]
    model = MODELS[trained["model"]](L=trained.get("attn_dim", 128)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    man = pd.read_csv(MANIFEST)
    grade_of = {Path(f).stem: g for f, g in zip(man["svs_file"], man["max_grade"])}

    rows = []
    for split in args.splits:
        loader = DataLoader(
            WSIBagDataset(
                split,
                standardize=trained.get("standardize", False),
                label_col="grade_ord" if trained["model"] == "ordinal" else "y",
            ),
            batch_size=1, shuffle=False, num_workers=args.num_workers,
        )
        for feats, _, name in loader:
            out = model(feats.to(device))
            prob, A = out[0].item(), out[2].squeeze(0).cpu().numpy()
            k = len(A)
            top5 = np.sort(A)[::-1][: max(1, int(0.05 * k))].sum()
            rows.append({
                "name": name[0], "grade": grade_of[name[0]], "prob": prob,
                "entropy": float(-(A * np.log(A + 1e-12)).sum() / np.log(k)),
                "top5pct": float(top5), "K": k,
            })

    df = pd.DataFrame(rows)
    print(f"체크포인트 {args.checkpoint}  (model={trained['model']}, "
          f"standardize={trained.get('standardize', False)}, lr={trained['lr']})")
    print(f"대상 split {args.splits}  총 {len(df)}장\n")

    normal = df[df.grade == "normal"]
    print(f"=== 정상 {len(normal)}장 대비 등급별 AUC ===")
    for g in GRADES[1:]:
        pos = df[df.grade == g]
        if len(pos) < 3:
            print(f"  정상 vs {g:<9} n={len(pos):>3}   (표본 부족, 생략)")
            continue
        labels = np.r_[np.zeros(len(normal)), np.ones(len(pos))]
        scores = np.r_[normal["prob"].to_numpy(), pos["prob"].to_numpy()]
        lo, hi = bootstrap_auc(labels, scores)
        print(f"  정상 vs {g:<9} n={len(pos):>3}   AUC {roc_auc_score(labels, scores):.4f}   "
              f"95% [{lo:.3f}, {hi:.3f}]")

    print(f"\n=== attention 집중도 ===")
    print(f"{'등급':<10}{'n':>5}{'엔트로피':>11}{'상위5%비중':>12}{'평균확률':>10}")
    for g in GRADES:
        sub = df[df.grade == g]
        if not len(sub):
            continue
        print(f"{g:<10}{len(sub):>5}{sub.entropy.mean():>11.4f}"
              f"{sub.top5pct.mean():>12.4f}{sub.prob.mean():>10.4f}")

    focal = df[df.grade.isin(["moderate", "severe"])]
    mild = df[df.grade.isin(["minimal", "slight"])]
    if len(focal) >= 3 and len(mild) >= 3:
        from scipy import stats
        t = stats.ttest_ind(focal.entropy, mild.entropy, equal_var=False)
        direction = "더 분산" if focal.entropy.mean() > mild.entropy.mean() else "더 집중"
        print(f"\n  moderate+severe(n={len(focal)})가 minimal+slight(n={len(mild)})보다 "
              f"{direction}  엔트로피 {focal.entropy.mean():.4f} vs {mild.entropy.mean():.4f}  "
              f"p={t.pvalue:.4f}")

    out = args.checkpoint.parent / "by_grade.csv"
    df.to_csv(out, index=False)
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()
