#!/usr/bin/env python3
"""
TG-GATEs .svs 전수 배율/해상도 점검 스크립트  (atlantic 서버에서 실행)

준비:
    pip install openslide-python pandas
    # 시스템 패키지도 필요할 수 있음:  sudo apt-get install -y openslide-tools

실행:
    python3 probe_svs_magnification.py
결과:
    svs_magnification_report.csv  (파일별 배율/mpp/크기/예상 패치수)
    + 콘솔에 요약(배율 분포, 이상치)
"""
import os, glob, sys
import pandas as pd

try:
    import openslide
except ImportError:
    sys.exit("openslide-python 이 없습니다.  pip install openslide-python 후 다시 실행하세요.")

# ── 설정 ─────────────────────────────────────────────
ROOT = "/atlantic_data/shj21/ABMIL/dataset/wsi"   # normal/ abnormal/ 상위. 옮겼으면 새 경로로 수정
PATCH = 256        # 20x 기준 패치 픽셀
TARGET_MAG = 20    # 목표 배율
# ─────────────────────────────────────────────────────

paths = sorted(glob.glob(os.path.join(ROOT, "**", "*.svs"), recursive=True))
print(f"찾은 .svs: {len(paths)}장\n")
if not paths:
    sys.exit(f"경로에 .svs가 없습니다: {ROOT}  (ROOT를 실제 위치로 고치세요)")

rows = []
for i, p in enumerate(paths, 1):
    rec = {"svs_file": os.path.basename(p),
           "label": os.path.basename(os.path.dirname(p))}
    try:
        s = openslide.OpenSlide(p)
        pr = s.properties
        W, H = s.dimensions
        mag = pr.get("aperio.AppMag") or pr.get("openslide.objective-power")
        mppx = pr.get("openslide.mpp-x")
        rec.update({
            "width": W, "height": H,
            "AppMag": mag,
            "mpp_x": mppx,
            "mpp_y": pr.get("openslide.mpp-y"),
            "vendor": pr.get("openslide.vendor"),
            "n_levels": s.level_count,
        })
        # 목표 20x로 맞추려면 필요한 resize factor
        try:
            rec["resize_to_20x"] = round(TARGET_MAG / float(mag), 4) if mag else None
        except (TypeError, ValueError):
            rec["resize_to_20x"] = None
        # 20x 기준 배경포함 최대 패치 격자수 (실제론 조직만 세면 더 적음)
        try:
            rf = rec["resize_to_20x"] or 1
            w20, h20 = W * rf, H * rf
            rec["approx_patches_20x"] = int(w20 // PATCH) * int(h20 // PATCH)
        except Exception:
            rec["approx_patches_20x"] = None
        s.close()
    except Exception as e:
        rec["error"] = str(e)
    rows.append(rec)
    if i % 100 == 0:
        print(f"  ...{i}/{len(paths)} 처리")

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svs_magnification_report.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")

print("\n===== 요약 =====")
print("총 파일:", len(df))
if "error" in df.columns:
    print("열기 실패:", df["error"].notna().sum(), "(있으면 report의 error 열 확인)")
print("\n[배율(AppMag) 분포]")
print(df["AppMag"].value_counts(dropna=False).to_string())
if "mpp_x" in df:
    mx = pd.to_numeric(df["mpp_x"], errors="coerce")
    print(f"\n[mpp-x] 최소 {mx.min():.4f} / 중앙 {mx.median():.4f} / 최대 {mx.max():.4f}  (20x≈0.5, 40x≈0.25)")
if "approx_patches_20x" in df:
    pp = pd.to_numeric(df["approx_patches_20x"], errors="coerce")
    print(f"[예상 패치수(배경포함)] 중앙 {int(pp.median())} / 최소 {int(pp.min())} / 최대 {int(pp.max())}")
print(f"\n저장: {out}")
