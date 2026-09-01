# -*- coding: utf-8 -*-
"""배포용 청크 압축 — processed/chunks/*.jsonl → *.jsonl.gz

청크 원본은 1.5GB라 저장소에 넣을 수 없다. gzip으로 약 16%(240MB 안팎)까지
줄어들고, retrieval.index_for()가 .jsonl이 없으면 .jsonl.gz를 읽는다.

  python baseline/pack_chunks.py            압축(원본 유지)
  python baseline/pack_chunks.py --check    압축본 존재·크기만 확인
"""
import gzip
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "processed" / "chunks"
LEVEL = 6          # 6이면 속도와 크기가 균형. 9는 20% 느리고 2% 작다.


def main():
    check = "--check" in sys.argv
    if not CHUNK_DIR.exists():
        sys.exit(f"청크 디렉터리 없음: {CHUNK_DIR}")
    src = sorted(CHUNK_DIR.glob("*.jsonl"))
    raw_total = gz_total = 0
    for i, p in enumerate(src, 1):
        out = p.with_suffix(".jsonl.gz")
        raw = p.stat().st_size
        if not check and (not out.exists() or out.stat().st_mtime < p.stat().st_mtime):
            with p.open("rb") as fi, gzip.open(out, "wb", compresslevel=LEVEL) as fo:
                shutil.copyfileobj(fi, fo, length=1 << 20)
        gz = out.stat().st_size if out.exists() else 0
        raw_total += raw
        gz_total += gz
        print(f"[{i:>2}/{len(src)}] {p.stem:<20} {raw / 1e6:>7.1f}MB → {gz / 1e6:>6.1f}MB")
    if raw_total:
        print(f"\n합계 {raw_total / 1e9:.2f}GB → {gz_total / 1e6:.0f}MB "
              f"({gz_total / raw_total:.0%})")
    missing = [p.stem for p in src if not p.with_suffix(".jsonl.gz").exists()]
    if missing:
        print("압축본 없음:", ", ".join(missing))


if __name__ == "__main__":
    main()
