# VIENCE-Inc 인턴십 작업 기록 (2026.07.01 – 2026.08.31)

두 달간의 인턴십 산출물입니다. 크게 두 갈래입니다.

1. **ABMIL 학습 모델 구축** — 독성병리 전슬라이드영상(WSI)의 병변 유무 분류
2. **PreTox mitosis detection** — 유사분열 검출 모델 개발

## 1. ABMIL — WSI 독성병리 슬라이드 분류

Attention-based Multiple Instance Learning으로 WSI의 병변 유무를 분류합니다.
TRIDENT로 조직 세그멘테이션·패치 추출을 하고, CONCH v1.5로 패치 특징(768차원)을
뽑은 뒤 ABMIL 헤드를 학습합니다.

## 파이프라인

```
dataset/wsi/  ──TRIDENT─→  contours_geojson/, patches/  ──CONCH v1.5─→  features_conch_v15/
                                                                              │
                                                          scripts/train.py ───┘
```

## 디렉터리

| 경로 | 내용 |
|---|---|
| `scripts/` | 학습·평가·전처리 코드 (`train.py`, `model.py`, `dataset.py`, `evaluate.py` 등) |
| `logs/` | 실험별 학습 로그·메트릭·체크포인트 (`best.pt`, `best_auc.pt`) |
| `metadata/`, `manifest/` | 슬라이드 메타데이터·매니페스트 CSV |
| `dataset/features/` | 세그멘테이션 결과, 패치 좌표, 썸네일, 실행 설정·상태 JSON |
| `PreTox/` | mitosis detection 관련 코드·모델 |
| `TRIDENT/`, `CONCH/`, `AttentionDeepMIL/` | 외부 저장소 스냅샷 (아래 참조) |

## 저장소에 포함되지 않은 데이터

GitHub 용량 제한으로 아래는 제외되어 있습니다. 원본은 작업 서버
`/atlantic_data/shj21/ABMIL/` 에 있습니다.

| 경로 | 크기 | 재생성 방법 |
|---|---|---|
| `dataset/wsi/` | 682 GB | 원본 SVS. `scripts/download.py` + `manifest/manifest.csv` |
| `dataset/features/20x_512px_0px_overlap/features_conch_v15/` | 12 GB | `wsi/` 에서 TRIDENT + CONCH v1.5 로 재추출 |
| `PreTox/mitosis_detection/wsi/` | 8.8 GB | `PreTox/mitosis_detection/download.sh` |
| `download_svs/*.svs` | 485 MB | 위와 동일 |
| `.venv/`, `PreTox/mitosis-venv/` | 3.8 GB | 가상환경. 아래 참조 |
| `.env` | — | DB 접속정보·시크릿. 별도 전달 |

특징 파일이 없어도 배치보정 통계(`slide_sums.npz`, `feature_stats_train.npz`)와
전처리 설정(`_config_*.json`)은 포함되어 있어 실험 설정 재현은 가능합니다.

### 사내 저장소 제외

인턴십 기간 중 PathoView 관련 개발도 진행했으나, 해당 코드베이스인
`pathoview/`(smart-path-viewer-model-deploy)와
`PreTox/PathoView/`(smart-path-viewer-backend)는 VIENCE-Inc 자산이므로
이 개인 저장소에는 포함하지 않았습니다. 각 사내 저장소에서 직접 확인하세요.

## 실행 환경

conda 환경 `abmil-env` 를 사용합니다. `.venv/` 는 이 프로젝트의 실제 환경이
아니므로 참고하지 마세요.

```bash
/home/shj21/miniconda3/envs/abmil-env/bin/python scripts/train.py --run-name base
```

## 외부 저장소 스냅샷

아래 디렉터리는 클론을 일반 파일로 포함한 것이라 자체 git 이력이 없습니다.
스냅샷 시점은 다음과 같습니다.

| 디렉터리 | 업스트림 | 브랜치 | 커밋 |
|---|---|---|---|
| `TRIDENT/` | https://github.com/mahmoodlab/TRIDENT | `main` | `457756c` |
| `CONCH/` | https://github.com/mahmoodlab/CONCH | `main` | `141cc09` |
| `AttentionDeepMIL/` | https://github.com/AMLab-Amsterdam/AttentionDeepMIL | `master` | `eb0434b` |
