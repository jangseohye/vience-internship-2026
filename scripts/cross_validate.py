"""화합물 단위 5-fold 교차검증으로 설정 간 차이를 신뢰구간과 함께 비교한다.

단일 split(val 212장)에서는 AUC의 95% 신뢰구간 폭이 0.145라 설정 간 차이(0.06)를
가려낼 수 없다. 교차검증으로 818장 전부를 평가에 쓰면 구간이 절반 이하로 줄어든다.

test split(202장, 화합물 27종)은 건드리지 않는다. 교차검증은 train+val(818장,
화합물 85종)에서만 수행한다. 여기에 test를 섞으면 최종 평가 대상이 오염된다.

fold 하나가 평가용이고, 그 다음 fold가 early stopping용 val, 나머지 셋이 train이다.
슬라이드마다 정확히 한 번씩 평가되므로 예측을 모아 하나의 AUC를 낼 수 있다.

    python scripts/cross_validate.py --config abmil_raw  --device cuda:0
    python scripts/cross_validate.py --config abmil_std  --device cuda:1
    python scripts/cross_validate.py --config meanpool_lr            # CPU
    python scripts/cross_validate.py --summary
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import WSIBagDataset, FEATURES_DIR, MANIFEST
from model import MODELS
from train import evaluate, set_seed, train_one_epoch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logs"
SLIDE_STATS = FEATURES_DIR.parent / "slide_sums.npz"
N_FOLDS = 5

CONFIGS = {
    # 비교 대상은 셋으로 좁힌다. 노이즈에 묻히지 않을 만큼 성격이 다른 것들이다.
    "meanpool_lr": {"kind": "lr", "standardize": True},
    "abmil_raw": {"kind": "abmil", "standardize": False, "lr": 1e-3, "reg": 1e-4},
    "abmil_std": {"kind": "abmil", "standardize": True, "lr": 5e-4, "reg": 1e-4},
}


def build_folds(df: pd.DataFrame, group_key: str = "COMPOUND_NAME") -> dict:
    """묶음 단위를 슬라이드 수 기준으로 균형 있게 5개 fold에 배정한다."""
    sizes = df.groupby(group_key, sort=False).size()
    # 큰 묶음부터 가장 비어 있는 fold에 넣으면 fold 간 슬라이드 수가 고르게 맞는다.
    order = sizes.sort_values(ascending=False, kind="stable").index
    load = [0] * N_FOLDS
    fold_of = {}
    for compound in order:
        f = int(np.argmin(load))
        fold_of[compound] = f
        load[f] += sizes[compound]
    return fold_of


def cache_slide_sums(df: pd.DataFrame) -> dict:
    """슬라이드별 특징 합·제곱합·개수를 한 번만 계산해 캐시한다.

    fold마다 train 통계를 다시 구해야 하는데, 매번 원본을 읽으면 37GB를 다섯 번
    읽게 된다. 슬라이드 단위 합을 미리 구해두면 fold 통계는 그것들을 더하기만 하면 된다.
    """
    if SLIDE_STATS.exists():
        z = np.load(SLIDE_STATS, allow_pickle=True)
        return {"names": list(z["names"]), "sum": z["sum"], "sumsq": z["sumsq"], "n": z["n"]}

    names, sums, sumsqs, ns = [], [], [], []
    for fname in df["svs_file"]:
        stem = Path(fname).stem
        with h5py.File(FEATURES_DIR / f"{stem}.h5", "r") as f:
            arr = f["features"][:].astype(np.float64)
        names.append(stem)
        sums.append(arr.sum(0))
        sumsqs.append((arr ** 2).sum(0))
        ns.append(arr.shape[0])
    out = {"names": names, "sum": np.array(sums), "sumsq": np.array(sumsqs), "n": np.array(ns)}
    np.savez(SLIDE_STATS, **out)
    print(f"슬라이드 통계 캐시 생성: {SLIDE_STATS}")
    return out


def fold_stats(cache: dict, train_stems: list) -> tuple:
    idx = [cache["names"].index(s) for s in train_stems]
    total = cache["sum"][idx].sum(0)
    total_sq = cache["sumsq"][idx].sum(0)
    n = cache["n"][idx].sum()
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def pooled_means(cache: dict, stems: list, stats) -> np.ndarray:
    """로지스틱 회귀용 평균 pooling 벡터. 표준화는 fold train 통계로 한다."""
    idx = [cache["names"].index(s) for s in stems]
    m = cache["sum"][idx] / cache["n"][idx][:, None]
    return ((m - stats[0]) / stats[1]).astype(np.float32)


@torch.no_grad()
def predict(model, loader, device) -> tuple:
    model.eval()
    probs, labels, names = [], [], []
    for feats, label, name in loader:
        prob = model(feats.to(device))[0]
        probs.append(prob.item())
        labels.append(int(label.item() > 0))
        names.append(name[0])
    return np.array(probs), np.array(labels), names


def run_abmil(cfg, sub, fold_col, stats, device, args) -> tuple:
    set_seed(args.seed)
    kw = dict(manifest=sub, split_col=fold_col, standardize=cfg["standardize"], stats=stats)
    loaders = {
        name: DataLoader(
            WSIBagDataset(name, **kw),
            batch_size=1,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for name in ("train", "val", "eval")
    }

    model = MODELS["attention"]().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg["reg"]
    )

    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, loaders["train"], optimizer, device)
        va = evaluate(model, loaders["val"], device)
        if va["loss"] < best_loss:
            best_loss, best_epoch, stale = va["loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    probs, labels, names = predict(model, loaders["eval"], device)
    return probs, labels, names, best_epoch


def run_lr(cfg, sub, fold_col, stats, cache) -> tuple:
    tr = sub[sub[fold_col] == "train"]
    ev = sub[sub[fold_col] == "eval"]
    tr_stems = [Path(f).stem for f in tr["svs_file"]]
    ev_stems = [Path(f).stem for f in ev["svs_file"]]
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(
        pooled_means(cache, tr_stems, stats), tr["y"].to_numpy()
    )
    probs = clf.predict_proba(pooled_means(cache, ev_stems, stats))[:, 1]
    return probs, ev["y"].to_numpy(), ev_stems, 0


def bootstrap_ci(labels, probs, n_boot=2000, seed=0) -> tuple:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[i])) > 1:
            vals.append(roc_auc_score(labels[i], probs[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize() -> None:
    rows = []
    for path in sorted(LOG_ROOT.glob("cv_*.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        print("교차검증 결과가 없다.")
        return
    rows.sort(key=lambda r: -r["auc"])
    print(f"{'config':<16}{'fold기준':<16}{'n':>6}{'AUC':>9}{'95% 구간':>20}{'fold별 AUC':>34}")
    print("-" * 101)
    for r in rows:
        ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        folds = " ".join(f"{a:.3f}" for a in r["fold_aucs"])
        print(f"{r['config']:<16}{r.get('group_by', 'COMPOUND_NAME'):<16}{r['n']:>6}"
              f"{r['auc']:>9.4f}{ci:>20}{folds:>34}")
    print("\n구간이 겹치면 두 설정은 구별되지 않는다.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=list(CONFIGS))
    p.add_argument(
        "--group-by",
        choices=("COMPOUND_NAME", "pair_id"),
        default="COMPOUND_NAME",
        help="fold를 나누는 단위. pair_id는 같은 화합물이 train과 eval에 함께 나타나므로 "
             "누수가 있는 대조군이다. 두 결과의 차이가 곧 '화합물을 외워서 얻는 이득'이다",
    )
    p.add_argument("--summary", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if args.summary:
        summarize()
        return
    if not args.config:
        p.error("--config 또는 --summary 중 하나가 필요하다")

    cfg = CONFIGS[args.config]
    device = args.device if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(MANIFEST)
    sub = df[df["split"].isin(["train", "val"])].reset_index(drop=True).copy()
    fold_of = build_folds(sub, args.group_by)
    sub["fold"] = sub[args.group_by].map(fold_of)

    print(f"config {args.config}  {cfg}")
    print(f"fold 기준 {args.group_by}  ({sub[args.group_by].nunique()}개 묶음)")
    print(f"교차검증 대상 {len(sub)}장, 화합물 {sub.COMPOUND_NAME.nunique()}종 "
          f"(test {int((df.split == 'test').sum())}장은 제외)")
    counts = sub.groupby("fold").agg(n=("svs_file", "size"), compounds=("COMPOUND_NAME", "nunique"))
    print("  fold별 슬라이드/화합물: " + ", ".join(
        f"f{i}={r.n}/{r.compounds}" for i, r in counts.iterrows()))

    cache = cache_slide_sums(sub)

    all_probs, all_labels, all_names, fold_aucs = [], [], [], []
    for f in range(N_FOLDS):
        col = f"cv{f}"
        sub[col] = np.where(
            sub["fold"] == f, "eval",
            np.where(sub["fold"] == (f + 1) % N_FOLDS, "val", "train"),
        )
        tr_stems = [Path(x).stem for x in sub.loc[sub[col] == "train", "svs_file"]]
        stats = fold_stats(cache, tr_stems) if cfg["standardize"] else None

        if cfg["kind"] == "abmil":
            probs, labels, names, best_ep = run_abmil(cfg, sub, col, stats, device, args)
        else:
            probs, labels, names, best_ep = run_lr(cfg, sub, col, stats, cache)

        auc = roc_auc_score(labels, probs)
        fold_aucs.append(float(auc))
        all_probs.append(probs); all_labels.append(labels); all_names += names
        print(f"  fold {f}  eval {len(labels):>3}장  AUC {auc:.4f}  (best epoch {best_ep})")

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = float(roc_auc_score(labels, probs))
    ci = bootstrap_ci(labels, probs)
    print(f"\n{args.config}  전체 {len(labels)}장  AUC {auc:.4f}  "
          f"95% 구간 [{ci[0]:.3f}, {ci[1]:.3f}]  폭 {ci[1] - ci[0]:.3f}")

    suffix = "" if args.group_by == "COMPOUND_NAME" else "_pair"
    out = LOG_ROOT / f"cv_{args.config}{suffix}.json"
    out.write_text(json.dumps({
        "config": args.config, "group_by": args.group_by, "settings": cfg, "n": len(labels),
        "auc": auc, "ci": list(ci), "fold_aucs": fold_aucs,
        "predictions": {n: float(p) for n, p in zip(all_names, probs)},
    }, indent=2))
    print(f"기록: {out}")


if __name__ == "__main__":
    main()
