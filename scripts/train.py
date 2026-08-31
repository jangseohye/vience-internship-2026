"""ABMIL 학습 — early stopping + best 체크포인트로 test 1회.

epoch마다 train → val 평가를 돌고 val_loss를 기록한다. val_loss가 patience 동안
개선되지 않으면 학습을 멈추고, val_loss가 가장 낮았던 시점의 가중치를 되살려
test를 딱 한 번만 수행한다. test는 학습 중 어떤 결정에도 관여하지 않는다.

    python scripts/train.py
    python scripts/train.py --model gated_attention --patience 15
"""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import WSIBagDataset
from model import MODELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODELS), default="attention")
    p.add_argument("--attn-dim", type=int, default=128, help="attention 병목 차원 L")
    p.add_argument("--epochs", type=int, default=200, help="최대 epoch. early stopping이 먼저 끊는 게 정상")
    p.add_argument("--lr", type=float, default=5e-4, help="원논문 기본값")
    p.add_argument("--reg", type=float, default=1e-4, help="weight decay. 원논문 기본값")
    p.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adam",
        help="adam은 원논문 설정. adamw는 weight decay를 gradient와 분리해 적용한다",
    )
    p.add_argument("--patience", type=int, default=20, help="val_loss가 이만큼 개선 없으면 중단")
    p.add_argument(
        "--standardize",
        action="store_true",
        help="train 통계로 특징을 차원별 표준화한다. 공통 성분이 신호를 가리는 문제를 완화",
    )
    p.add_argument(
        "--skip-test",
        action="store_true",
        help="test를 건너뛴다. 하이퍼파라미터 탐색 중에는 test 숫자를 보면 안 되므로 이 옵션을 쓴다",
    )
    p.add_argument("--min-delta", type=float, default=0.0, help="개선으로 인정할 최소 감소폭")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-name", default=None, help="기본값은 시각 기반 이름")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split: str, args: argparse.Namespace) -> DataLoader:
    # 슬라이드마다 패치 수가 달라 bag을 묶을 수 없다. batch_size는 항상 1.
    return DataLoader(
        WSIBagDataset(
            split,
            standardize=args.standardize,
            label_col="grade_ord" if args.model == "ordinal" else "y",
        ),
        batch_size=1,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=True,
    )


def train_one_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    total_loss = 0.0
    probs, labels = [], []

    for feats, label, _ in loader:
        feats, label = feats.to(device), label.to(device)

        optimizer.zero_grad()
        loss, _ = model.calculate_objective(feats, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            prob = model(feats)[0]
        probs.append(prob.item())
        # ordinal은 label이 등급 인덱스다. 지표는 이진 기준으로 재서 기존 실행과 비교 가능하게 한다.
        labels.append(int(label.item() > 0))

    return metrics(total_loss / len(loader), probs, labels)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    probs, labels = [], []

    for feats, label, _ in loader:
        feats, label = feats.to(device), label.to(device)
        loss, _ = model.calculate_objective(feats, label)
        prob = model(feats)[0]

        total_loss += loss.item()
        probs.append(prob.item())
        labels.append(int(label.item() > 0))

    return metrics(total_loss / len(loader), probs, labels)


def metrics(loss: float, probs: list, labels: list) -> dict:
    probs_a, labels_a = np.array(probs), np.array(labels)
    preds = (probs_a >= 0.5).astype(int)
    return {
        "loss": loss,
        "acc": float((preds == labels_a).mean()),
        # 라벨이 한 종류뿐이면 AUC가 정의되지 않는다 (split이 1:1이라 실제로는 발생하지 않음)
        "auc": float(roc_auc_score(labels_a, probs_a)) if len(np.unique(labels_a)) > 1 else float("nan"),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = LOG_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    loaders = {s: make_loader(s, args) for s in ("train", "val", "test")}

    model = MODELS[args.model](L=args.attn_dim).to(device)
    optim_cls = optim.Adam if args.optimizer == "adam" else optim.AdamW
    optimizer = optim_cls(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg
    )

    print(f"run       : {run_dir}")
    print(f"device    : {device}")
    print(f"model     : {args.model}  L={args.attn_dim}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"optim     : {args.optimizer}  lr={args.lr}  weight_decay={args.reg}")
    print(f"standardize: {args.standardize}")
    print(f"slides    : train {len(loaders['train'].dataset)} / "
          f"val {len(loaders['val'].dataset)} / test {len(loaders['test'].dataset)}")
    print(f"early stop: patience={args.patience}, min_delta={args.min_delta}\n")

    history_path = run_dir / "history.csv"
    ckpt_path = run_dir / "best.pt"
    best_val_loss = float("inf")
    best_val = None          # 최저 시점의 val 지표 전체. 하이퍼파라미터 비교에 쓴다
    best_epoch = 0
    epochs_no_improve = 0

    # val_loss와 val_auc는 어긋날 수 있다. 모델이 과신하면 loss는 나빠지는데 순위
    # 능력인 AUC는 유지되거나 오른다. early stopping은 val_loss로 하되(기존 실행과의
    # 비교를 위해), AUC 최고 시점도 따로 저장해 두 기준을 나중에 비교할 수 있게 한다.
    auc_ckpt_path = run_dir / "best_auc.pt"
    best_val_auc = -1.0
    best_auc_epoch = 0

    with open(history_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc", "best"]
        )

        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, loaders["train"], optimizer, device)
            va = evaluate(model, loaders["val"], device)

            improved = va["loss"] < best_val_loss - args.min_delta
            if improved:
                best_val_loss = va["loss"]
                best_val = va
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_loss": va["loss"],
                        "args": vars(args),
                    },
                    ckpt_path,
                )
            else:
                epochs_no_improve += 1

            if va["auc"] > best_val_auc:
                best_val_auc = va["auc"]
                best_auc_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_loss": va["loss"],
                        "val_auc": va["auc"],
                        "args": vars(args),
                    },
                    auc_ckpt_path,
                )

            writer.writerow([
                epoch,
                f"{tr['loss']:.6f}", f"{tr['acc']:.4f}", f"{tr['auc']:.4f}",
                f"{va['loss']:.6f}", f"{va['acc']:.4f}", f"{va['auc']:.4f}",
                int(improved),
            ])
            fh.flush()

            print(
                f"epoch {epoch:>3}  "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} auc {tr['auc']:.4f}  |  "
                f"val loss {va['loss']:.4f} acc {va['acc']:.4f} auc {va['auc']:.4f}"
                f"{'  ← best' if improved else f'  ({epochs_no_improve}/{args.patience})'}"
            )

            if epochs_no_improve >= args.patience:
                print(f"\nearly stopping: val_loss가 {args.patience} epoch 동안 개선되지 않음")
                break

    print(
        f"\nval_loss 기준 best epoch {best_epoch}  "
        f"loss {best_val['loss']:.4f} acc {best_val['acc']:.4f} auc {best_val['auc']:.4f}"
    )
    print(f"val_auc  기준 best epoch {best_auc_epoch}  auc {best_val_auc:.4f}  ({auc_ckpt_path.name})")

    # 학습이 완전히 끝난 뒤에야 test를 한 번 본다.
    # 하이퍼파라미터를 고르는 중이라면 --skip-test로 이 숫자를 아예 보지 않는다.
    if args.skip_test:
        te = None
        print("--skip-test 지정: test를 수행하지 않음")
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        te = evaluate(model, loaders["test"], device)
        print(f"TEST  loss {te['loss']:.4f}  acc {te['acc']:.4f}  auc {te['auc']:.4f}")

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_auc_epoch": best_auc_epoch,
        "best_val_auc": best_val_auc,
        "test": te,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n기록: {history_path}\n      {run_dir / 'summary.json'}\n      {ckpt_path}")


if __name__ == "__main__":
    main()
