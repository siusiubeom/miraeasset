"""코퍼스 전체 → 회사별 종합 마크다운 빌드.

산출물 (C:\\mirae\\processed\\):
  docs/<doc_group>/<corp_name>/<doc_id>.md   문서 단위 마크다운 (원문 보존)
  companies/<listed_name>.md                 회사당 1개 종합 파일
                                             (헤더 + 문서목록 + 정정로그 + 전체 본문, 접수일순)
  build_report.json                          빌드 통계·실패·정정 매칭 결과

사용법: python build_company_md.py [--limit N] [--company 회사명]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dart_to_md import convert_file, read_text

CORPUS = Path(r"C:\mirae\corpus")
OUT = Path(r"C:\mirae\processed")

_DOCNAME_RE = re.compile(r"<DOCUMENT-NAME[^>]*>([^<]+)</DOCUMENT-NAME>", re.I)
# 정정본 본문에서 "정정관련(대상) 공시서류 제출일" 필드 탐색
_CORR_FIELD_RE = re.compile(r"정정\s*(?:관련|대상)\s*공시\s*서류(?:명)?\s*의?\s*(?:최초\s*)?제출일")
_DATE_RE = re.compile(r"(20\d{2})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")


def _doc_files(folder: Path, rcept_no: str) -> list[Path]:
    """본문 파일 먼저, 첨부(감사보고서 등)는 뒤로."""
    xmls = sorted(folder.glob("*.xml"))
    main = [f for f in xmls if f.stem == rcept_no]
    rest = [f for f in xmls if f.stem != rcept_no]
    return main + rest


def _extract_corr_targets(md_head: str) -> list[str]:
    """변환된 본문 앞부분에서 정정 대상 제출일(YYYYMMDD)들을 추출."""
    targets = []
    for m in _CORR_FIELD_RE.finditer(md_head):
        window = md_head[m.end() : m.end() + 150]
        for d in _DATE_RE.finditer(window):
            y, mo, dd = d.groups()
            targets.append(f"{y}{int(mo):02d}{int(dd):02d}")
    seen = set()
    return [t for t in targets if not (t in seen or seen.add(t))]


def convert_doc(rec: dict) -> dict:
    """한 문서(폴더)를 마크다운으로 변환해 저장. Pool 워커에서 실행."""
    doc_id = rec["doc_id"]
    folder = CORPUS / rec["file_path"]
    out_path = OUT / "docs" / rec["doc_group"] / rec["corp_name"] / f"{doc_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"doc_id": doc_id, "out": str(out_path), "in_bytes": 0, "out_bytes": 0,
              "corr_targets": [], "error": None, "note": None}
    try:
        parts = []
        if rec["file_format"] == "pdf+html":
            # 뷰어 HTML을 같은 파서로 시도 (원문 XML 미제공 3건)
            htmls = sorted(folder.glob("*.html"))
            result["in_bytes"] = sum(f.stat().st_size for f in folder.iterdir())
            for f in htmls:
                md = convert_file(f)
                if len(md) > 500:
                    parts.append(md)
                    result["note"] = "viewer.html에서 변환 (원문 XML 미제공)"
            if not parts:
                pdfs = [f.name for f in folder.glob("*.pdf")]
                parts.append(f"> ⚠️ 이 문서는 원문 XML이 제공되지 않아 본문 변환에 실패했습니다. "
                             f"원본 PDF: {', '.join(pdfs)} (경로: {rec['file_path']}) — 별도 처리 필요\n")
                result["note"] = "변환 실패 — PDF 별도 처리 필요"
        else:
            files = _doc_files(folder, rec["rcept_no"])
            result["in_bytes"] = sum(f.stat().st_size for f in files)
            for i, f in enumerate(files):
                md = convert_file(f)
                if i > 0:
                    name_m = _DOCNAME_RE.search(read_text(f)[:2000])
                    name = name_m.group(1).strip() if name_m else f.name
                    parts.append(f"\n\n---\n\n# [첨부문서] {name}\n\n{md}")
                else:
                    parts.append(md)
        body = "".join(parts)
        if rec["is_correction"]:
            result["corr_targets"] = _extract_corr_targets(body[:8000])
        header = (f"<!-- DOC id={doc_id} group={rec['doc_group']} subtype={rec.get('doc_subtype') or '-'} "
                  f"rcept_no={rec['rcept_no']} rcept_dt={rec['rcept_dt']} "
                  f"correction={'Y' if rec['is_correction'] else 'N'} corp={rec['corp_name']} -->\n"
                  f"# 【{rec['listed_name']}】 {rec['report_nm']} — 접수 {rec['rcept_dt']} ({rec['rcept_no']})"
                  f"{' 〔정정본〕' if rec['is_correction'] else ''}\n\n")
        out_path.write_text(header + body, encoding="utf-8")
        result["out_bytes"] = out_path.stat().st_size
    except Exception as e:  # noqa: BLE001 — 개별 실패는 리포트로 수집
        result["error"] = f"{type(e).__name__}: {e}"
    return result


_GROUP_KO = {"periodic": "정기", "major": "주요사항", "exchange": "거래소", "holding": "지분"}


def build_company_file(uni: dict, docs: list[dict], results: dict[str, dict],
                       corr_matches: dict[str, list], superseded: dict[str, list]) -> Path:
    """회사당 1개 종합 md 조립."""
    name = uni["listed_name"]
    safe = re.sub(r'[\\/:*?"<>|]', "_", name)
    path = OUT / "companies" / f"{safe}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = sorted(docs, key=lambda r: (r["rcept_dt"], r["rcept_no"]))
    n_corr = sum(1 for d in docs if d["is_correction"])
    cap = int(uni["market_cap"] or 0)

    head = []
    head.append(f"<!-- COMPANY {name} stock={uni['stock_code']} corp_code={uni['corp_code']} -->")
    head.append(f"# {name} — 공시 종합 문서")
    head.append("")
    head.append(f"- 법인명: {uni['corp_name']} | 영문명: {uni['corp_eng_name']} | 종목코드: {uni['stock_code']} | 시장: {uni['market']}")
    head.append(f"- 업종: {uni['industry']} | 섹터: {uni['sector']} | 상장일: {uni['listing_date']} | 결산월: {uni['fiscal_month']}")
    head.append(f"- 시가총액: {cap:,}억원 (2026-07-24 기준)")
    head.append(f"- 수록 문서: {len(docs)}건 (정기 {uni['n_periodic']} / 주요사항 {uni['n_major']} / 거래소 {uni['n_exchange']} / 지분 {uni['n_holding']}) — 이 중 정정본 {n_corr}건")
    head.append("")
    head.append("## 문서 목록 (접수일순)")
    head.append("")
    head.append("| # | 접수일 | 유형 | 보고서명 | 접수번호 | 비고 |")
    head.append("|---|---|---|---|---|---|")
    for i, d in enumerate(docs, 1):
        notes = []
        if d["is_correction"]:
            notes.append("정정본")
        if d["rcept_no"] in superseded:
            notes.append("→ 후속 정정 있음: " + ", ".join(superseded[d["rcept_no"]]))
        r = results.get(d["doc_id"], {})
        if r.get("note"):
            notes.append(r["note"])
        if r.get("error"):
            notes.append("변환실패")
        dt = d["rcept_dt"]
        head.append(f"| {i} | {dt[:4]}-{dt[4:6]}-{dt[6:]} | {_GROUP_KO.get(d['doc_group'], d['doc_group'])} "
                    f"| {d['report_nm']} | {d['rcept_no']} | {'; '.join(notes)} |")
    head.append("")
    head.append("## 정정 로그")
    head.append("")
    corr_docs = [d for d in docs if d["is_correction"]]
    if not corr_docs:
        head.append("(정정 공시 없음)")
    for d in corr_docs:
        m = corr_matches.get(d["doc_id"], {})
        targets = m.get("targets", [])
        matched = m.get("matched", [])
        dt = d["rcept_dt"]
        line = f"- {dt[:4]}-{dt[4:6]}-{dt[6:]} **{d['report_nm']}** ({d['rcept_no']})"
        if matched:
            line += " → 정정 대상: " + ", ".join(f"{x['report_nm']} ({x['rcept_no']}, {x['rcept_dt']})" for x in matched)
        elif targets:
            line += f" → 대상 제출일 {', '.join(targets)} — **코퍼스에 원본 없음(수집기간 밖) 또는 미확인**"
        else:
            line += " → 대상 제출일 추출 실패 (본문 확인 필요)"
        head.append(line)
    head.append("")
    head.append("> ⚠️ 정정본이 있는 문서의 수치는 폐기된 값일 수 있음 — 항상 최신 정정본을 우선할 것.")
    head.append("")

    with path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(head))
        for d in docs:
            r = results.get(d["doc_id"], {})
            fp.write("\n\n" + "=" * 80 + "\n\n")
            if r.get("out") and not r.get("error"):
                fp.write(Path(r["out"]).read_text(encoding="utf-8"))
            else:
                fp.write(f"<!-- DOC id={d['doc_id']} 변환실패 -->\n# 【{name}】 {d['report_nm']} — 변환 실패: {r.get('error')}\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="처리할 문서 수 제한 (테스트용)")
    ap.add_argument("--company", default=None, help="특정 회사만 (corp_name)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in (CORPUS / "manifest.jsonl").open(encoding="utf-8")]
    universe = {r["corp_name"]: r for r in csv.DictReader((CORPUS / "universe.csv").open(encoding="utf-8-sig"))}

    todo = manifest
    if args.company:
        todo = [r for r in todo if r["corp_name"] == args.company]
    if args.limit:
        todo = todo[: args.limit]

    t0 = time.time()
    print(f"convert: {len(todo)} docs, workers={args.workers}", flush=True)
    results: dict[str, dict] = {}
    with mp.Pool(args.workers) as pool:
        for n, res in enumerate(pool.imap_unordered(convert_doc, todo, chunksize=4), 1):
            results[res["doc_id"]] = res
            if n % 200 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

    # 정정 매칭: 같은 회사+같은 그룹에서 대상 제출일과 접수일이 일치하는 문서
    by_corp_group: dict[tuple, list[dict]] = {}
    rec_by_id = {r["doc_id"]: r for r in todo}
    for r in todo:
        by_corp_group.setdefault((r["corp_code"], r["doc_group"]), []).append(r)
    corr_matches: dict[str, dict] = {}
    superseded: dict[str, list[str]] = {}
    for r in todo:
        if not r["is_correction"]:
            continue
        targets = results.get(r["doc_id"], {}).get("corr_targets", [])
        cands = [c for c in by_corp_group.get((r["corp_code"], r["doc_group"]), [])
                 if c["rcept_dt"] in targets and c["rcept_no"] < r["rcept_no"]]
        corr_matches[r["doc_id"]] = {
            "targets": targets,
            "matched": [{"doc_id": c["doc_id"], "report_nm": c["report_nm"],
                         "rcept_no": c["rcept_no"], "rcept_dt": c["rcept_dt"]} for c in cands],
        }
        for c in cands:
            superseded.setdefault(c["rcept_no"], []).append(r["rcept_no"])

    # 회사별 종합 파일
    by_corp: dict[str, list[dict]] = {}
    for r in todo:
        by_corp.setdefault(r["corp_name"], []).append(r)
    print(f"assemble: {len(by_corp)} companies", flush=True)
    company_sizes = {}
    for corp, docs in sorted(by_corp.items()):
        uni = universe[corp]
        p = build_company_file(uni, docs, results, corr_matches, superseded)
        company_sizes[uni["listed_name"]] = p.stat().st_size

    errors = {k: v["error"] for k, v in results.items() if v["error"]}
    notes = {k: v["note"] for k, v in results.items() if v["note"]}
    n_corr = sum(1 for r in todo if r["is_correction"])
    n_matched = sum(1 for v in corr_matches.values() if v["matched"])
    n_target_only = sum(1 for v in corr_matches.values() if v["targets"] and not v["matched"])
    n_no_target = sum(1 for v in corr_matches.values() if not v["targets"])
    report = {
        "docs_total": len(todo),
        "in_bytes": sum(v["in_bytes"] for v in results.values()),
        "out_bytes": sum(v["out_bytes"] for v in results.values()),
        "elapsed_sec": round(time.time() - t0, 1),
        "corrections": {"total": n_corr, "matched": n_matched,
                        "target_found_but_no_original": n_target_only,
                        "target_extraction_failed": n_no_target},
        "errors": errors,
        "notes": notes,
        "company_file_bytes": company_sizes,
        "corr_matches": corr_matches,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done in {report['elapsed_sec']}s | in {report['in_bytes']/1e9:.2f}GB -> out {report['out_bytes']/1e9:.2f}GB")
    print(f"corrections: {n_corr} total / {n_matched} matched / {n_target_only} original-missing / {n_no_target} no-target")
    if errors:
        print(f"ERRORS: {len(errors)}")
        for k, v in list(errors.items())[:10]:
            print("  ", k, v)


if __name__ == "__main__":
    main()
