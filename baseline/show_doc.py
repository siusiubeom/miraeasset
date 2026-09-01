# -*- coding: utf-8 -*-
"""청크 원문 열람 — 문항 작성용.

BM25를 타지 않고 processed/chunks/<회사>.jsonl 을 직접 읽는다. 검색 품질과
무관하게 "그 문서에 무엇이 적혀 있는가"를 그대로 보기 위한 도구다.

  python baseline/show_doc.py 20241118000328
      접수번호의 모든 청크를 section_path 순으로 출력(manifest 메타 헤더 포함)

  python baseline/show_doc.py --company 에코프로비엠 --grep 전환가액
      그 회사 청크 중 문자열이 든 것만 출력(접수번호·section_path 표시)

  옵션: --chars 2000  청크당 출력 길이 (0이면 전문)
        --context 80  --grep 일치 지점 앞뒤로만 보기
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "processed" / "chunks"
MANIFEST = ROOT / "corpus" / "manifest.jsonl"

DEFAULT_CHARS = 2000
RULE = "─" * 78


def load_manifest():
    if not MANIFEST.exists():
        return {}
    out = {}
    for ln in MANIFEST.open(encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out[r["rcept_no"]] = r
    return out


def chunk_files(company=None):
    """대상 청크 파일 목록. 회사 지정 시 부분일치로 고른다."""
    if not CHUNK_DIR.exists():
        sys.exit("청크 디렉터리 없음: " + str(CHUNK_DIR))
    files = sorted(CHUNK_DIR.glob("*.jsonl"))
    if not company:
        return files
    key = company.strip()
    hit = [f for f in files if key == f.stem or key in f.stem]
    if not hit:
        sys.exit("회사 청크 파일 없음: {} (예: {})".format(
            company, ", ".join(f.stem for f in files[:5])))
    return hit


def iter_chunks(files):
    for f in files:
        for ln in f.open(encoding="utf-8"):
            if ln.strip():
                yield json.loads(ln)


def fmt_dt(d):
    return (d[:4] + "-" + d[4:6] + "-" + d[6:]) if d and len(d) == 8 else (d or "")


def print_meta(rcept_no, meta, n_chunks):
    m = meta.get(rcept_no)
    print(RULE)
    if not m:
        print("접수번호 {}  (manifest에 메타 없음)".format(rcept_no))
    else:
        print("{} | {} | 접수 {} | 접수번호 {}".format(
            m["listed_name"], m["report_nm"], fmt_dt(m["rcept_dt"]), rcept_no))
        base = ("{}년 {}월".format(m["base_year"], m["base_month"])
                if m.get("base_year") else "-")
        print("  doc_group={} subtype={} 정정={} 기준={} 파일={}".format(
            m["doc_group"], m.get("doc_subtype") or "-",
            "예" if m.get("is_correction") else "아니오", base,
            m.get("file_format", "")))
        print("  경로: " + str(m.get("file_path", "")))
    print("  청크 {}개".format(n_chunks))
    print(RULE)


def body(text):
    """청크 첫 줄은 출처 헤더다. 본문만 돌려준다."""
    parts = (text or "").split("\n", 1)
    return parts[1].strip() if len(parts) == 2 else (text or "").strip()


def windows(text, needle, context):
    """일치 지점 앞뒤 context 글자씩. 겹치는 구간은 하나로 합친다."""
    spans = []
    i = text.find(needle)
    while i >= 0:
        start, end = max(0, i - context), min(len(text), i + len(needle) + context)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
        i = text.find(needle, i + len(needle))
    return [text[a:b].replace("\n", " ") for a, b in spans]


def clip(s, chars):
    if not chars or len(s) <= chars:
        return s
    return s[:chars] + "\n  … (이하 {}자 생략, --chars 0 으로 전문)".format(len(s) - chars)


def show_by_rcept(rcept_no, chars):
    meta = load_manifest()
    m = meta.get(rcept_no)
    # 회사를 알면 그 파일만, 모르면 전체를 훑는다.
    files = chunk_files(m["listed_name"]) if m else chunk_files()
    rows = [c for c in iter_chunks(files) if c.get("rcept_no") == rcept_no]
    if not rows:
        sys.exit("해당 접수번호의 청크 없음: " + rcept_no)
    rows.sort(key=lambda c: (c.get("section_path") or "", c.get("chunk_id") or ""))
    print_meta(rcept_no, meta, len(rows))
    for c in rows:
        print("\n[{}] {}".format(c["chunk_id"], c.get("section_path") or "(절 경로 없음)"))
        print(clip(body(c.get("text", "")), chars) or "  (본문 없음 — 출처 헤더만)")
    return rows


def show_grep(company, needle, chars, context):
    meta = load_manifest()
    files = chunk_files(company)
    hits = [c for c in iter_chunks(files) if needle in (c.get("text") or "")]
    scope = company if company else "전체 회사"
    print("{} 청크 중 '{}' 포함 — {}건 / 문서 {}개".format(
        scope, needle, len(hits), len({c["rcept_no"] for c in hits})))
    if not hits:
        return hits
    hits.sort(key=lambda c: (c.get("rcept_dt") or "", c.get("rcept_no") or "",
                             c.get("chunk_id") or ""))
    for c in hits:
        corr = " [정정]" if c.get("correction") else ""
        print("\n" + RULE)
        print("{} | {} | 접수 {} | {}{}".format(
            c.get("corp_name", ""), c.get("report_nm", ""),
            fmt_dt(c.get("rcept_dt", "")), c.get("rcept_no", ""), corr))
        print("  절: {}".format(c.get("section_path") or "(없음)"))
        print("  청크: {}".format(c.get("chunk_id")))
        text = body(c.get("text", ""))
        if context:
            for w in windows(text, needle, context):
                print("  … " + w + " …")
        else:
            print(clip(text, chars) or "  (본문 없음 — 출처 헤더만)")
    return hits


def main():
    ap = argparse.ArgumentParser(description="청크 원문 열람 — 문항 작성용")
    ap.add_argument("rcept_no", nargs="?", help="접수번호 14자리")
    ap.add_argument("--company", help="회사명(부분일치)")
    ap.add_argument("--grep", help="청크 텍스트 부분일치 검색어")
    ap.add_argument("--chars", type=int, default=DEFAULT_CHARS,
                    help="청크당 출력 길이 (0=전문)")
    ap.add_argument("--context", type=int, default=0,
                    help="--grep 일치 지점 앞뒤 글자 수만 출력")
    args = ap.parse_args()

    if args.rcept_no:
        show_by_rcept(args.rcept_no.strip(), args.chars)
    elif args.grep:
        show_grep(args.company, args.grep, args.chars, args.context)
    else:
        ap.error("접수번호 또는 --grep 중 하나는 필요하다")


if __name__ == "__main__":
    main()
