"""저장된 체크포인트를 재학습 없이 평가한다.

하이퍼파라미터 탐색은 --skip-test로 돌리므로, 설정을 확정한 뒤 그 실행의 best.pt를
이 스크립트로 test에 한 번 적용한다. 재학습하면 GPU 비결정성 때문에 "val이 가장
좋았던 그 모델"과 미세하게 달라질 수 있다. 체크포인트를 직접 쓰면 그 여지가 없다.

    python scripts/evaluate.py logs/base/best.pt
    python scripts/evaluate.py logs/base/best.pt --split val    # 검산용
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import WSIBagDataset
from model import MODELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="train.py가 저장한 best.pt")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save", action="store_true", help="체크포인트 옆에 결과 json을 남긴다")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trained = ckpt["args"]

    # 학습 때와 똑같은 구조로 되살린다. 인자가 어긋나면 state_dict 로드가 실패한다.
    model = MODELS[trained["model"]](L=trained.get("attn_dim", 128)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"checkpoint : {args.checkpoint}")
    print(f"  학습 설정 : {trained['model']}  L={trained.get('attn_dim', 128)}  "
          f"{trained.get('optimizer', 'adam')}  lr={trained['lr']}  wd={trained['reg']}  "
          f"standardize={trained.get('standardize', False)}")
    print(f"  best epoch: {ckpt['epoch']}  (val_loss {ckpt['val_loss']:.4f})")

    # 학습 때와 같은 전처리를 써야 한다. 다르면 평가가 무의미해진다.
    loader = DataLoader(
        WSIBagDataset(
            args.split,
            standardize=trained.get("standardize", False),
            label_col="grade_ord" if trained["model"] == "ordinal" else "y",
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    total_loss = 0.0
    probs, labels, names = [], [], []
    for feats, label, name in loader:
        feats, label = feats.to(device), label.to(device)
        loss, _ = model.calculate_objective(feats, label)
        prob = model(feats)[0]
        total_loss += loss.item()
        probs.append(prob.item())
        labels.append(int(label.item() > 0))
        names.append(name[0])

    probs_a, labels_a = np.array(probs), np.array(labels)
    preds = (probs_a >= 0.5).astype(int)
    result = {
        "split": args.split,
        "n": len(labels_a),
        "loss": total_loss / len(loader),
        "acc": float((preds == labels_a).mean()),
        "auc": float(roc_auc_score(labels_a, probs_a)),
    }
    print(
        f"\n{args.split.upper()}  n={result['n']}  "
        f"loss {result['loss']:.4f}  acc {result['acc']:.4f}  auc {result['auc']:.4f}"
    )

    # 어떤 슬라이드를 틀렸는지 남겨두면 이후 분석에 쓸 수 있다.
    wrong = [(n, int(y), round(p, 4)) for n, y, p in zip(names, labels_a, probs_a) if (p >= 0.5) != y]
    print(f"오분류 {len(wrong)}장 (예: {wrong[:5]})")

    if args.save:
        out = args.checkpoint.parent / f"eval_{args.split}.json"
        out.write_text(json.dumps({**result, "misclassified": wrong}, indent=2))
        print(f"기록: {out}")


if __name__ == "__main__":
    main()
