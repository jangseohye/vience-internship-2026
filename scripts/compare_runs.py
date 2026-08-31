"""logs/ 아래의 학습 실행들을 val 성능 기준으로 비교한다.

하이퍼파라미터는 반드시 val로 고른다. test 열은 이미 확정된 설정을 한 번 평가한
결과일 뿐이며, 이 값을 보고 설정을 고르면 test가 두 번째 validation이 되어 최종
성능이 부풀려진다.

    python scripts/compare_runs.py
    python scripts/compare_runs.py --sort auc
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=LOG_ROOT)
    parser.add_argument("--sort", choices=("loss", "auc"), default="loss", help="val 기준 정렬 키")
    args = parser.parse_args()

    runs = []
    for path in sorted(args.logs.glob("*/summary.json")):
        runs.append(json.loads(path.read_text()))
    if not runs:
        print(f"비교할 실행이 없다: {args.logs}/*/summary.json")
        return

    runs.sort(
        key=lambda r: r["best_val"]["loss"] if args.sort == "loss" else -r["best_val"]["auc"]
    )

    header = (
        f"{'run':<22}{'model':<16}{'L':>5}{'lr':>9}{'wd':>9}"
        f"{'ep*':>5}{'val_loss':>10}{'val_acc':>9}{'val_auc':>9}{'test_auc':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        a, v = r["args"], r["best_val"]
        test_auc = f"{r['test']['auc']:.4f}" if r.get("test") else "-"
        print(
            f"{r['run_name']:<22}{a['model']:<16}{a.get('attn_dim', 128):>5}"
            f"{a['lr']:>9.0e}{a['reg']:>9.0e}{r['best_epoch']:>5}"
            f"{v['loss']:>10.4f}{v['acc']:>9.4f}{v['auc']:>9.4f}{test_auc:>10}"
        )

    best = runs[0]
    print(f"\nval {args.sort} 기준 1위: {best['run_name']}")
    if any(r.get("test") for r in runs) and len(runs) > 1:
        print(
            "주의: test 열이 채워진 실행이 여러 개다. 설정 선택은 val로만 하고,\n"
            "      test는 최종 설정 하나에 대해서만 보는 것이 원칙이다."
        )


if __name__ == "__main__":
    main()
