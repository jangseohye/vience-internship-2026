"""AttentionDeepMIL(Ilse et al., 2018)을 CONCH v1.5 특징에 맞게 포팅.

원본(AttentionDeepMIL/model.py)은 MNIST-bags용이라 LeNet-5 특징 추출기가 붙어 있고
M=500이다. 여기서는 CONCH v1.5가 이미 인코딩을 끝냈으므로 특징 추출기를 제거하고
M=768로 맞춘다. attention과 classifier 구조는 원본 그대로다.

입력은 (n_patches, 768). DataLoader가 batch_size=1로 앞에 차원을 하나 붙이므로
forward에서 squeeze(0)으로 떼어낸다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 768


class Attention(nn.Module):
    """원논문의 기본 attention pooling."""

    def __init__(self, M: int = FEATURE_DIM, L: int = 128, attention_branches: int = 1) -> None:
        super().__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attention_branches

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),                     # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES),    # vector w
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        H = x.squeeze(0)                  # (K, M) — K는 패치 수, 슬라이드마다 다름

        A = self.attention(H)             # (K, ATTENTION_BRANCHES)
        A = torch.transpose(A, 1, 0)      # (ATTENTION_BRANCHES, K)
        A = F.softmax(A, dim=1)           # 패치 전체에 대해 정규화

        Z = torch.mm(A, H)                # (ATTENTION_BRANCHES, M) — 가중평균

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))
        return neg_log_likelihood, A


class GatedAttention(nn.Module):
    """gated 변형. 원논문에서 기본형보다 근소하게 우수하다고 보고됨."""

    def __init__(self, M: int = FEATURE_DIM, L: int = 128, attention_branches: int = 1) -> None:
        super().__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attention_branches

        self.attention_V = nn.Sequential(nn.Linear(self.M, self.L), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.M, self.L), nn.Sigmoid())
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        H = x.squeeze(0)                  # (K, M)

        A_V = self.attention_V(H)         # (K, L)
        A_U = self.attention_U(H)         # (K, L)
        A = self.attention_w(A_V * A_U)   # (K, ATTENTION_BRANCHES)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        Z = torch.mm(A, H)

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))
        return neg_log_likelihood, A


class OrdinalAttention(Attention):
    """중증도 등급을 순서형으로 학습하는 변형.

    normal < minimal < slight < moderate < severe 라는 순서를 손실에 반영한다.
    이진 라벨은 minimal과 severe를 똑같이 1로 취급해 정보를 버리는데, 슬라이드가
    606장뿐인 상황에서 그 손실이 크다.

    누적 이진 분류(cumulative link)로 구현한다. K개 등급이면 K-1개의 출력이
    각각 P(등급 > k)를 뜻한다. attention 부분은 원본과 동일하다.
    """

    def __init__(self, M: int = FEATURE_DIM, L: int = 128, attention_branches: int = 1,
                 n_grades: int = 5) -> None:
        super().__init__(M=M, L=L, attention_branches=attention_branches)
        self.n_grades = n_grades
        # Sigmoid를 층에 넣지 않는다. 손실 쪽에서 logit을 그대로 받는 편이 수치적으로 안정적이다.
        self.classifier = nn.Linear(self.M * self.ATTENTION_BRANCHES, n_grades - 1)

    def forward(self, x):
        H = x.squeeze(0)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, H)

        logits = self.classifier(Z)                  # (1, K-1)
        cum = torch.sigmoid(logits)                  # P(등급 > k), k=0..K-2
        Y_prob = cum[:, :1]                          # P(등급 > normal) = abnormal 확률
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A, logits

    def calculate_objective(self, X, Y):
        """Y는 등급 인덱스(0=normal … 4=severe)."""
        _, _, A, logits = self.forward(X)
        # 등급 g는 k < g 인 모든 문턱을 넘는다. 목표는 [1,1,...,0,0] 꼴.
        k = torch.arange(self.n_grades - 1, device=logits.device)
        target = (Y.view(-1, 1).float() > k.view(1, -1)).float()
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        return loss, A

    def calculate_classification_error(self, X, Y):
        # 이진(정상/이상) 기준으로 잰다. 기존 실행들과 비교 가능하게 하기 위함이다.
        Y_bin = (Y > 0).float()
        Y_prob, _, _, _ = self.forward(X)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return 1.0 - Y_hat.eq(Y_bin).cpu().float().mean().item(), Y_hat


MODELS = {
    "attention": Attention,
    "gated_attention": GatedAttention,
    "ordinal": OrdinalAttention,
}


if __name__ == "__main__":
    for name, cls in MODELS.items():
        model = cls()
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, 3868, FEATURE_DIM)          # 패치 3868개짜리 슬라이드 1장
        y = torch.tensor([3] if name == "ordinal" else [1])   # ordinal은 등급 인덱스
        out = model(x)
        prob, hat, A = out[0], out[1], out[2]
        loss, _ = model.calculate_objective(x, y)
        print(
            f"{name:<16} params={n_params:>8,}  "
            f"Y_prob={prob.item():.4f}  A={tuple(A.shape)}  "
            f"attn합={A.sum().item():.4f}  loss={loss.item():.4f}"
        )
