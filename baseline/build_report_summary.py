# -*- coding: utf-8 -*-
"""전사 빌드 산출물 점검 리포트.

build_report.json + processed/chunks 를 읽어 코퍼스 상태를 요약한다.
실행: python baseline/build_report_summary.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "processed" / "build_report.json"
CHUNKS = ROOT / "processed" / "chunks"
MANIFEST = ROOT / "corpus" / "manifest.jsonl"


def main():
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    corp_of = {}
    for line in MANIFEST.open(encoding="utf-8"):
        d = json.loads(line)
        corp_of[d["doc_id"]] = d["listed_name"]

    print(f"docs_total={rep['docs_total']}  elapsed={rep['elapsed_sec']}s  "
          f"in={rep['in_bytes'] / 1e9:.2f}GB -> out={rep['out_bytes'] / 1e9:.2f}GB")

    # 1) 정정 매칭 집계
    c = rep["corrections"]
    print("\n[1] corrections")
    for k, v in c.items():
        print(f"  {k}: {v}")
    orphan = {k: v for k, v in rep["corr_matches"].items()
              if v["targets"] and not v["matched"]}
    print(f"\n  target_found_but_no_original 상세 ({len(orphan)}건) "
          f"— 정정본은 있으나 원본이 코퍼스 밖 (T6 한계고지 소재)")
    by_corp = Counter(corp_of.get(k, "?") for k in orphan)
    for corp, n in by_corp.most_common():
        print(f"    {corp}: {n}건")
    for k, v in list(orphan.items())[:10]:
        print(f"      {k} ({corp_of.get(k, '?')}) targets={v['targets']}")

    nt = {k: v for k, v in rep["corr_matches"].items() if not v["targets"]}
    if nt:
        print(f"\n  target_extraction_failed 상세 ({len(nt)}건) — 본문에서 정정 대상일 미추출")
        for corp, n in Counter(corp_of.get(k, "?") for k in nt).most_common(10):
            print(f"    {corp}: {n}건")

    # 2) errors
    print(f"\n[2] errors ({len(rep['errors'])}건)")
    for k, v in list(rep["errors"].items())[:20]:
        print(f"  {k} ({corp_of.get(k, '?')}): {str(v)[:120]}")

    # 3) notes
    print(f"\n[3] notes ({len(rep['notes'])}건)")
    for k, v in list(rep["notes"].items())[:20]:
        print(f"  {k} ({corp_of.get(k, '?')}): {str(v)[:160]}")

    # 4) 회사별 청크 수
    if CHUNKS.exists():
        counts = {}
        for f in CHUNKS.glob("*.jsonl"):
            with f.open(encoding="utf-8") as fp:
                counts[f.stem] = sum(1 for _ in fp)
        order = sorted(counts.items(), key=lambda x: -x[1])
        print(f"\n[4] 회사별 청크 수 ({len(counts)}개사, 총 {sum(counts.values()):,})")
        print("  상위 5:", ", ".join(f"{k} {v:,}" for k, v in order[:5]))
        print("  하위 5:", ", ".join(f"{k} {v:,}" for k, v in order[-5:]))
    else:
        print("\n[4] chunks 디렉터리 없음 — chunk_docs.py 미실행")

    # 5) 정정 체인이 잡힌 회사 수
    chain_corps = {corp_of.get(k, "?") for k, v in rep["corr_matches"].items() if v["matched"]}
    print(f"\n[5] 정정 체인이 잡힌 회사: {len(chain_corps)}/70")
    print("  ", ", ".join(sorted(chain_corps)[:25]) + (" ..." if len(chain_corps) > 25 else ""))


if __name__ == "__main__":
    main()
