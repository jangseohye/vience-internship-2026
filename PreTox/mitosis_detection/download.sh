#!/bin/bash
# Open TG-GATEs "Increased mitosis" (severe/moderate) WSI 14장 다운로드
# 이어받기(resume) 지원 — 중단 후 다시 실행하면 남은 부분만 받습니다.
#   실행: setsid nohup bash download.sh >/dev/null 2>&1 &
DEST=/atlantic_data/shj21/ABMIL/PreTox/mitosis_detection/wsi
CSV=/atlantic_data/shj21/ABMIL/PreTox/increased_mitosis_severe_moderate.csv
LOG=/atlantic_data/shj21/ABMIL/PreTox/mitosis_detection/download.log
LIST=/atlantic_data/shj21/ABMIL/PreTox/mitosis_detection/urls.tsv
mkdir -p "$DEST"

# 중복 실행 방지 — 같은 파일에 두 프로세스가 쓰면 파일이 깨집니다
exec 9>"$DEST/.download.lock"
flock -n 9 || { echo "[$(date +%H:%M:%S)] 이미 실행 중 — 종료" >> "$LOG"; exit 0; }

python3 -c "
import csv,sys
for r in csv.DictReader(open('$CSV',encoding='utf-8-sig')):
    print(r['svs_file'],r['url'],sep='\t')
" > "$LIST"

get_one() {
  local name="$1" url="$2" dest="$3" log="$4"
  local exp cur
  exp=$(curl -sIL --max-time 60 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  cur=$(stat -c %s "$dest/$name" 2>/dev/null || echo 0)
  if [ -n "$exp" ] && [ "$cur" = "$exp" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name (이미 완료)" >> "$log"; return 0
  fi
  for try in 1 2 3 4 5; do
    curl -sSL -C - --max-time 7200 -o "$dest/$name" "$url" 2>>"$log"
    cur=$(stat -c %s "$dest/$name" 2>/dev/null || echo 0)
    if [ -n "$exp" ] && [ "$cur" = "$exp" ]; then
      echo "[$(date +%H:%M:%S)] OK   $name ($((cur/1024/1024)) MB)" >> "$log"; return 0
    fi
    echo "[$(date +%H:%M:%S)] RETRY $name ($try) $cur/$exp" >> "$log"; sleep 5
  done
  echo "[$(date +%H:%M:%S)] FAIL $name" >> "$log"; return 1
}
export -f get_one

echo "[$(date +%H:%M:%S)] === 다운로드 시작 (14장) ===" >> "$LOG"
# 동시 3개
xargs -a "$LIST" -d '\n' -P 3 -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r n u <<< "{}"; get_one "$n" "$u" "'"$DEST"'" "'"$LOG"'"'

n=$(ls -1 "$DEST"/*.svs 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] === 완료: $n/14장, 총 $(du -sh "$DEST" | cut -f1) ===" >> "$LOG"
