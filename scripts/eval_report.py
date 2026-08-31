"""이진 분류 성능 리포트 — 지표 전체 + 민감도 제약 임계값 최적화.

비임상 독성 스크리닝에서는 이상을 놓치는 비용이 정상을 오판하는 비용보다 훨씬 크다.
따라서 기본 임계값 0.5가 아니라 "민감도 >= 목표치를 만족하면서 특이도가 최대"인
임계값을 써야 한다. 이 스크립트는 두 지점을 나란히 놓고 비교한다.

    python scripts/eval_report.py logs/std_lr5e-4/best.pt
    python scripts/eval_report.py logs/std_lr5e-4/best.pt --split test --target-sens 0.95
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, average_precision_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import WSIBagDataset
from model import MODELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--target-sens", type=float, default=0.95, help="반드시 확보할 최소 민감도")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def predict(ckpt_path: Path, split: str, device: str, num_workers: int):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tr = ckpt["args"]
    model = MODELS[tr["model"]](L=tr.get("attn_dim", 128)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = WSIBagDataset(
        split,
        standardize=tr.get("standardize", False),
        label_col="grade_ord" if tr["model"] == "ordinal" else "y",
    )
    probs, labels, names = [], [], []
    for feats, label, name in DataLoader(ds, batch_size=1, num_workers=num_workers):
        probs.append(model(feats.to(device))[0].item())
        labels.append(int(label.item() > 0))   # 순서형이어도 지표는 이진 기준
        names.append(name[0])
    return np.array(probs), np.array(labels), names, tr, ckpt


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_acc": float((recall_score(y, pred, zero_division=0) + tn / (tn + fp)) / 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pick_threshold(y: np.ndarray, p: np.ndarray, target: float):
    """민감도 >= target을 만족하는 임계값 중 특이도가 가장 높은 것.

    roc_curve가 주는 후보만 훑으면 충분하다. 임계값을 그 사이로 옮겨도
    예측이 바뀌지 않으므로 (tpr, fpr) 조합은 이 유한 집합이 전부다.
    """
    fpr, tpr, thr = roc_curve(y, p)
    ok = tpr >= target
    if not ok.any():
        return None, fpr, tpr, thr
    # 조건을 만족하는 것 중 fpr 최소 = 특이도 최대. 동률이면 민감도가 높은 쪽.
    cand = np.where(ok)[0]
    best = cand[np.lexsort((-tpr[cand], fpr[cand]))[0]]
    return float(thr[best]), fpr, tpr, thr


def boot_ci(y, p, fn, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


ROWS = [
    ("정확도 (Accuracy)", "accuracy"), ("민감도 (Sensitivity)", "sensitivity"),
    ("특이도 (Specificity)", "specificity"), ("정밀도 (Precision/PPV)", "precision"),
    ("음성예측도 (NPV)", "npv"), ("F1-score", "f1"), ("균형정확도", "balanced_acc"),
]


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    p, y, names, tr, ckpt = predict(args.checkpoint, args.split, device, args.num_workers)

    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    lo, hi = boot_ci(y, p, roc_auc_score, args.n_boot)

    print("=" * 78)
    print(f"성능 리포트  |  {args.checkpoint}")
    print("=" * 78)
    print(f"모델      : {tr['model']}  L={tr.get('attn_dim',128)}  lr={tr['lr']}  "
          f"wd={tr['reg']}  standardize={tr.get('standardize', False)}")
    print(f"체크포인트: epoch {ckpt['epoch']} (val_loss {ckpt['val_loss']:.4f})")
    print(f"평가 대상 : {args.split}  n={len(y)}  (정상 {(y==0).sum()} / 이상 {(y==1).sum()})")
    print(f"\n[임계값 무관 지표]")
    print(f"  AUROC {auroc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   (0.5 = 무작위)")
    print(f"  AUPRC {auprc:.4f}   (기저율 {y.mean():.4f})")

    m05 = metrics_at(y, p, 0.5)
    thr, fpr, tpr, thrs = pick_threshold(y, p, args.target_sens)
    if thr is None:
        print(f"\n민감도 {args.target_sens:.0%}를 만족하는 임계값이 없다."); return
    mt = metrics_at(y, p, thr)

    # 참고용: Youden J 최적점 (민감도 제약 없음)
    j = int(np.argmax(tpr - fpr))
    mj = metrics_at(y, p, float(thrs[j]))

    print(f"\n[임계값별 비교]  목표 민감도 >= {args.target_sens:.0%}")
    w = 24
    print(f"{'지표':<{w}}{'기본 0.5':>12}{'민감도제약':>12}{'Youden J':>12}{'변화':>10}")
    print("-" * (w + 46))
    print(f"{'임계값':<{w}}{0.5:>12.4f}{thr:>12.4f}{thrs[j]:>12.4f}{'':>10}")
    for label, k in ROWS:
        d = mt[k] - m05[k]
        print(f"{label:<{w}}{m05[k]:>12.4f}{mt[k]:>12.4f}{mj[k]:>12.4f}{d:>+10.4f}")

    print(f"\n[혼동 행렬]  행=실제, 열=예측")
    for name, m in [("기본 임계값 0.5", m05), (f"민감도제약 {thr:.4f}", mt)]:
        print(f"\n  {name}")
        print(f"                예측:정상   예측:이상")
        print(f"    실제:정상   {m['tn']:>9}   {m['fp']:>9}")
        print(f"    실제:이상   {m['fn']:>9}   {m['tp']:>9}")
        print(f"    → 놓친 이상 {m['fn']}장, 불필요 정밀검사 {m['fp']}장")

    out = args.checkpoint.parent / f"report_{args.split}.json"
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "split": args.split, "n": len(y),
        "auroc": auroc, "auroc_ci": [lo, hi], "auprc": auprc,
        "at_0.5": m05, "at_target_sens": mt, "at_youden": mj,
        "target_sens": args.target_sens,
        "predictions": {n: float(v) for n, v in zip(names, p)},
    }, indent=2, ensure_ascii=False))
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()
