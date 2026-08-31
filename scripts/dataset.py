"""TRIDENT가 뽑은 CONCH v1.5 특징을 ABMIL 학습용으로 읽는다.

슬라이드 하나가 bag 하나다. 슬라이드마다 패치 수가 다르므로 배치로 묶을 수 없고,
DataLoader는 batch_size=1로만 쓴다.

    from dataset import WSIBagDataset
    ds = WSIBagDataset(split="train")
    feats, label, name = ds[0]        # (n_patches, 768), (), str
"""

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "manifest" / "manifest.csv"
FEATURES_DIR = (
    PROJECT_ROOT / "dataset" / "features" / "20x_512px_0px_overlap" / "features_conch_v15"
)
STATS_PATH = FEATURES_DIR.parent / "feature_stats_train.npz"
FEATURE_DIM = 768


class WSIBagDataset(Dataset):
    """manifest의 한 행 = 슬라이드 1장 = h5 1개 = bag 1개."""

    def __init__(
        self,
        split: str,
        manifest=MANIFEST,
        features_dir: Path = FEATURES_DIR,
        return_coords: bool = False,
        standardize: bool = False,
        stats_path: Path = STATS_PATH,
        split_col: str = "split",
        stats=None,
        label_col: str = "y",
    ) -> None:
        self.split = split
        self.features_dir = Path(features_dir)
        self.return_coords = return_coords

        # 차원별 표준화. 통계는 학습에 쓰이는 슬라이드에서만 계산해야 한다.
        # 평가 대상 슬라이드가 통계에 섞이면 그 자체가 누수다.
        # 교차검증에서는 fold마다 달라지므로 stats로 직접 넘길 수 있게 열어둔다.
        self.mean = self.std = None
        if standardize:
            if stats is None:
                if not Path(stats_path).exists():
                    raise FileNotFoundError(
                        f"통계 파일이 없다: {stats_path}\n"
                        "scripts/compute_feature_stats.py를 먼저 실행할 것."
                    )
                loaded = np.load(stats_path)
                stats = (loaded["mean"], loaded["std"])
            self.mean = torch.as_tensor(stats[0], dtype=torch.float32)
            self.std = torch.as_tensor(stats[1], dtype=torch.float32)

        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        if df[split_col].isna().any():
            raise ValueError(f"manifest의 {split_col}이 비어 있다. scripts/make_split.py를 먼저 실행할 것.")

        self.df = df[df[split_col] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"{split_col}={split!r}에 해당하는 행이 없다.")

        # h5 경로를 미리 만들어두고, 없는 파일은 지금 잡는다.
        # 학습 도중에 터지면 원인 추적이 훨씬 번거로워진다.
        self.paths = [self.features_dir / f"{Path(f).stem}.h5" for f in self.df["svs_file"]]
        missing = [p.name for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{split}: h5 {len(missing)}개 없음 (예: {missing[:5]}) — {self.features_dir}"
            )

        # label_col="grade_ord"면 0=normal … 4=severe인 순서형 라벨이 나온다.
        if label_col not in self.df.columns:
            raise KeyError(f"manifest에 {label_col} 컬럼이 없다. scripts/add_grade.py를 먼저 실행할 것.")
        self.labels = self.df[label_col].to_numpy(dtype=np.int64)
        self.binary = self.df["y"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        with h5py.File(self.paths[idx], "r") as f:
            feats = torch.from_numpy(f["features"][:])
            coords = torch.from_numpy(f["coords"][:]) if self.return_coords else None

        if self.mean is not None:
            feats = (feats - self.mean) / self.std

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        name = self.paths[idx].stem

        if self.return_coords:
            return feats, label, name, coords
        return feats, label, name

    def class_counts(self) -> dict:
        return {int(k): int(v) for k, v in zip(*np.unique(self.labels, return_counts=True))}


def _self_check() -> None:
    """모든 split을 훑어 shape과 라벨 분포를 확인한다."""
    for split in ("train", "val", "test"):
        ds = WSIBagDataset(split)
        feats, label, name = ds[0]
        counts = ds.class_counts()
        print(
            f"{split:<6} {len(ds):>4}장  "
            f"normal={counts.get(0, 0)} abnormal={counts.get(1, 0)}  "
            f"| 첫 샘플 {name}: {tuple(feats.shape)} {feats.dtype}, y={label.item()}"
        )
        if feats.shape[1] != FEATURE_DIM:
            raise ValueError(f"특징 차원이 {FEATURE_DIM}이 아니다: {feats.shape}")

    # 패치 수가 슬라이드마다 다른지 확인 (bag 크기 가변 = ABMIL의 전제)
    ds = WSIBagDataset("train")
    sizes = [ds[i][0].shape[0] for i in range(min(20, len(ds)))]
    print(f"\ntrain 앞 20장 패치 수: 최소 {min(sizes)}, 최대 {max(sizes)} — bag 크기 가변 확인")


if __name__ == "__main__":
    _self_check()
