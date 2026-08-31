#!/usr/bin/env python3
"""
TG-GATEs WSI 다운로더 (manifest 기반) — 경량판

이전 버전 대비 변경점:
  * 파일마다 열던 '크기 확인용 FTP 연결(ftplib)'을 제거 → 서버로 가는 연결 수 절반.
    전송 완결성 검증은 wget이 자체적으로 수행(서버가 알려주는 전체 크기만큼
    받아야 성공 처리, 미완이면 -c 로 이어받음).
  * 재개는 'wget -c(이어받기)'로 처리. 이미 완결된 파일은 wget이 재다운로드하지 않음.

동작 요약:
  - 재개(resume) : 이미 있고 완결된 파일은 wget이 건너뜀. 미완 파일은 이어받음.
  - 재시도(retry): 파일별 실패 시 백오프 두고 N회 재시도. 한 파일 실패가 전체를 멈추지 않음.
  - 병렬          : 스레드로 동시 다운로드(기본 2). 상대 FTP 동시연결 제한을 고려해 낮게.
  - 로그          : logs/download/ 에 실행 로그와 실패 목록(CSV).

사용 예:
  python3 download.py --limit 30 --workers 2     # 테스트(앞 30장, 동시 2)
  python3 download.py --workers 2                 # 전체
  python3 download.py --workers 2                 # 재실행 = 미완·실패분만 이어받기
"""
import argparse
import csv
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ----------------------------- 설정(기본값) -----------------------------
DEFAULT_ROOT = "/atlantic_data/shj21/ABMIL"      # PROJECT_ROOT (대소문자 주의: ABMIL)
DEFAULT_MANIFEST = None                          # 기본: <root>/manifest/manifest.csv
DEFAULT_WORKERS = 2                              # 상대 서버 동시연결 제한 고려, 낮게 시작
DEFAULT_RETRIES = 3
WGET_TIMEOUT = 180                               # 초, wget 연결/응답 타임아웃(대용량 고려 넉넉히)


def download_one(row: dict, root: Path, retries: int, log: logging.Logger):
    """한 파일 처리. (svs_file, status, reason) 반환. status in {ok, fail}."""
    svs = row["svs_file"]
    url = row["url"]
    local = root / row["file_path"]
    local.parent.mkdir(parents=True, exist_ok=True)

    last_reason = ""
    for attempt in range(1, retries + 1):
        try:
            # -c: 이어받기(이미 완결이면 재다운로드 안 함), -P: 대상 폴더에 원래 파일명으로 저장.
            # --tries=1 : wget 내부 재시도는 끄고, 재시도는 파이썬 루프가 관리(백오프 적용).
            proc = subprocess.run(
                ["wget", "-c", "-nv", "--tries=1",
                 f"--timeout={WGET_TIMEOUT}", "-P", str(local.parent), url],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            if proc.returncode == 0 and local.exists() and local.stat().st_size > 0:
                log.info(f"OK    {svs} ({local.stat().st_size} bytes)")
                return (svs, "ok", "")
            last_reason = f"wget rc={proc.returncode}"
            tail = (proc.stdout or "").strip().splitlines()
            if tail:
                last_reason += f" | {tail[-1][:120]}"
        except Exception as e:
            last_reason = f"예외:{e}"

        log.warning(f"RETRY {svs} ({attempt}/{retries}) - {last_reason}")
        time.sleep(min(5 * attempt, 30))     # 백오프: 5s, 10s, 15s ...

    log.error(f"FAIL  {svs} - {last_reason}")
    return (svs, "fail", last_reason)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="PROJECT_ROOT")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="manifest.csv 경로(기본: <root>/manifest/manifest.csv)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="동시 다운로드 수")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="파일별 재시도 횟수")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만(0=전체, 테스트용)")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest" / "manifest.csv"
    log_dir = root / "logs" / "download"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"download_{stamp}.log"
    fail_path = log_dir / f"failures_{stamp}.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("dl")

    if not manifest.exists():
        log.error(f"manifest를 찾을 수 없음: {manifest}")
        sys.exit(1)

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    log.info(f"시작: {len(rows)}개 | root={root} | workers={args.workers} | 로그={log_path}")
    t0 = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, r, root, args.retries, log): r for r in rows}
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(rows):
                log.info(f"진행 {done}/{len(rows)} | 경과 {(time.time()-t0)/60:.1f}분")

    ok = sum(1 for _, s, _ in results if s == "ok")
    fail = [(svs, reason) for svs, s, reason in results if s == "fail"]

    if fail:
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["svs_file", "reason"])
            w.writerows(fail)

    log.info("=" * 60)
    log.info(f"완료: 성공 {ok} | 실패 {len(fail)} | 총 {len(rows)}")
    log.info(f"총 소요 {(time.time()-t0)/60:.1f}분")
    if fail:
        log.info(f"실패 목록: {fail_path}  (같은 명령을 재실행하면 미완·실패분만 이어받음)")
    else:
        log.info("실패 없음. 모든 파일 다운로드 완료.")


if __name__ == "__main__":
    main()
