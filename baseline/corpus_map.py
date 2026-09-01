# -*- coding: utf-8 -*-
"""코퍼스 조회 도구 — 문제지 작성용.

corpus/manifest.jsonl 을 읽어 보고서명 분포·회사별 요약·정정 밀도를 내고,
CLI 로 보고서명/회사/기간 조회를 한다. 표준 라이브러리만 쓴다.

  python baseline/corpus_map.py                        요약 3종 + corpus_map.json
  python baseline/corpus_map.py --report 전환사채       보고서명 부분일치 회사·접수번호
  python baseline/corpus_map.py --company 에코프로비엠  회사 타임라인
  python baseline/corpus_map.py --company 에스엠 --from 20230201 --to 20230430
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.jsonl"
OUT_JSON = Path(__file__).resolve().parent / "corpus_map.json"

TOP_REPORTS = 40
TOP_CORRECTION = 15

# 보고서명 정규화 — 대괄호 표기([기재정정], [첨부정정] 등)와 기준월 괄호를 걷어낸다.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_PAREN_DATE_RE = re.compile(r"\s*\(\s*\d{4}\.\d{1,2}\s*\)")

# 문제지에서 소재로 쓸 만한 공시 종류 — 있는지 반드시 확인한다.
WATCH_REPORTS = (
    "전환사채권발행결정", "신주인수권부사채권발행결정",
    "교환사채권발행결정", "조건부자본증권발행결정",
    "유상증자결정", "무상증자결정",
    "회사분할결정", "회사합병결정",
    "채무보증결정", "타법인주식및출자증권취득결정",
    "소송등의판결·결정", "감자결정",
    "전환청구권행사", "신주인수권행사",
)


def normalize_report(nm):
    """보고서명에서 정정 표기와 기준월을 제거한 종류명."""
    s = _BRACKET_RE.sub("", nm or "")
    s = _PAREN_DATE_RE.sub("", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def watch_key(nm):
    """대조용 키 — 공백과 가운뎃점 변형을 흡수한다."""
    return re.sub(r"[\s·ㆍ・]", "", nm or "")


def load(path=MANIFEST):
    if not path.exists():
        sys.exit("manifest 없음: " + str(path))
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


def fmt_dt(d):
    return (d[:4] + "-" + d[4:6] + "-" + d[6:]) if d and len(d) == 8 else (d or "")


# ── [1] 보고서명 분포 ────────────────────────────────────────────────────────
def report_distribution(recs):
    counts = Counter(normalize_report(r["report_nm"]) for r in recs)
    corr = Counter(normalize_report(r["report_nm"])
                   for r in recs if r.get("is_correction"))
    rows = [{"report": nm, "n": n, "n_correction": corr.get(nm, 0)}
            for nm, n in counts.most_common()]

    # 관찰 대상은 부분일치로 찾는다 — 원문 표기가 조금씩 다르다.
    watch = []
    for target in WATCH_REPORTS:
        key = watch_key(target)
        names = sorted(nm for nm in counts if key in watch_key(nm))
        corps = sorted({r["listed_name"] for r in recs
                        if key in watch_key(normalize_report(r["report_nm"]))})
        watch.append({"target": target,
                      "n": sum(counts[nm] for nm in names),
                      "n_corps": len(corps),
                      "names": names[:5],
                      "corps": corps})
    return rows, watch


# ── [2] 회사별 요약 ──────────────────────────────────────────────────────────
def company_summary(recs):
    by_corp = defaultdict(list)
    for r in recs:
        by_corp[r["listed_name"]].append(r)
    out = []
    for name, rs in by_corp.items():
        dts = sorted(r["rcept_dt"] for r in rs if r.get("rcept_dt"))
        out.append({
            "listed_name": name,
            "n_docs": len(rs),
            "by_group": dict(sorted(Counter(r["doc_group"] for r in rs).items())),
            "n_correction": sum(1 for r in rs if r.get("is_correction")),
            "first_dt": dts[0] if dts else "",
            "last_dt": dts[-1] if dts else "",
        })
    return sorted(out, key=lambda d: -d["n_docs"])


# ── [3] 정정 밀도 ────────────────────────────────────────────────────────────
def correction_density(summary):
    rows = [dict(r, correction_pct=round(100 * r["n_correction"] / r["n_docs"], 1))
            for r in summary if r["n_docs"]]
    return sorted(rows, key=lambda d: (-d["correction_pct"], -d["n_docs"]))


# ── [4]~[6] 조회 ─────────────────────────────────────────────────────────────
def find_by_report(recs, needle):
    key = watch_key(needle)
    hits = [r for r in recs if key in watch_key(r["report_nm"])]
    by_corp = defaultdict(list)
    for r in sorted(hits, key=lambda r: (r["listed_name"], r["rcept_dt"])):
        by_corp[r["listed_name"]].append(r)
    return by_corp


def timeline(recs, company, dt_from="", dt_to=""):
    key = (company or "").strip()
    rows = [r for r in recs
            if key == r["listed_name"] or key in r["listed_name"] or key in r["corp_name"]]
    if dt_from:
        rows = [r for r in rows if r["rcept_dt"] >= dt_from]
    if dt_to:
        rows = [r for r in rows if r["rcept_dt"] <= dt_to]
    return sorted(rows, key=lambda r: (r["rcept_dt"], r["rcept_no"]))


# ── 출력 ─────────────────────────────────────────────────────────────────────
def print_overview(rows, watch, summary, density, n_total):
    print("[0] 문서 {}건, 회사 {}개, 보고서명 종류 {}종\n".format(
        n_total, len(summary), len(rows)))

    print("[1] 보고서명 분포 (상위 {}종)".format(TOP_REPORTS))
    print("  {:>5} {:>4}  보고서명".format("건수", "정정"))
    for r in rows[:TOP_REPORTS]:
        print("  {:>5} {:>4}  {}".format(r["n"], r["n_correction"], r["report"]))

    print("\n[1-1] 문제 소재 후보 보유 여부")
    for w in watch:
        mark = "O" if w["n"] else "-"
        line = "  {} {:<26} {:>4}건 / {:>2}개사".format(
            mark, w["target"], w["n"], w["n_corps"])
        if w["n"]:
            line += "  " + ", ".join(w["corps"][:5])
            if w["n_corps"] > 5:
                line += " 외 {}개사".format(w["n_corps"] - 5)
        print(line)
    missing = [w["target"] for w in watch if not w["n"]]
    if missing:
        print("  ※ 코퍼스에 없음: " + ", ".join(missing))

    print("\n[2] 회사별 요약 ({}개사)".format(len(summary)))
    print("  {:<18}{:>5}{:>5}  {:<12}{:<12}그룹별".format("회사", "문서", "정정", "최초", "최종"))
    for r in summary:
        groups = " ".join("{}:{}".format(k, v) for k, v in r["by_group"].items())
        print("  {:<18}{:>5}{:>5}  {:<12}{:<12}{}".format(
            r["listed_name"][:17], r["n_docs"], r["n_correction"],
            fmt_dt(r["first_dt"]), fmt_dt(r["last_dt"]), groups))

    print("\n[3] 정정 밀도 상위 {}개사 (정정 체인 문항 소재)".format(TOP_CORRECTION))
    print("  {:<18}{:>7}{:>5}{:>6}".format("회사", "정정%", "정정", "문서"))
    for r in density[:TOP_CORRECTION]:
        print("  {:<18}{:>7}{:>5}{:>6}".format(
            r["listed_name"][:17], r["correction_pct"], r["n_correction"], r["n_docs"]))


def print_report_hits(by_corp, needle):
    total = sum(len(v) for v in by_corp.values())
    print("[4] '{}' 부분일치 — {}건 / {}개사".format(needle, total, len(by_corp)))
    for name, rs in sorted(by_corp.items(), key=lambda kv: -len(kv[1])):
        print("\n  {} ({}건)".format(name, len(rs)))
        for r in rs:
            corr = " [정정]" if r.get("is_correction") else ""
            print("    {}  {}  {}{}".format(
                fmt_dt(r["rcept_dt"]), r["rcept_no"], r["report_nm"], corr))


def print_timeline(rows, company, dt_from, dt_to):
    span = ""
    if dt_from or dt_to:
        span = " {}~{}".format(fmt_dt(dt_from) or "처음", fmt_dt(dt_to) or "끝")
    print("[5] {} 타임라인{} — {}건".format(company, span, len(rows)))
    if not rows:
        print("  해당 조건의 공시 없음")
        return
    print("  {:<12}{:<10}{:<16}{:<5}보고서명".format("접수일", "유형", "접수번호", "정정"))
    for r in rows:
        print("  {:<12}{:<10}{:<16}{:<5}{}".format(
            fmt_dt(r["rcept_dt"]), r["doc_group"], r["rcept_no"],
            "정정" if r.get("is_correction") else "", r["report_nm"]))
    # 같은 날 여러 건은 정정 체인·다발 구간의 표시다.
    per_day = Counter(r["rcept_dt"] for r in rows)
    dup = sorted(d for d, n in per_day.items() if n > 1)
    if dup:
        print("\n  같은 날 2건 이상 접수: " + ", ".join(
            "{}({}건)".format(fmt_dt(d), per_day[d]) for d in dup))


def main():
    ap = argparse.ArgumentParser(description="코퍼스 조회 — 문제지 작성용")
    ap.add_argument("--report", help="보고서명 부분일치 조회")
    ap.add_argument("--company", help="회사 타임라인 조회")
    ap.add_argument("--from", dest="dt_from", default="", help="접수일 시작 (YYYYMMDD)")
    ap.add_argument("--to", dest="dt_to", default="", help="접수일 끝 (YYYYMMDD)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    recs = load(Path(args.manifest))
    rows, watch = report_distribution(recs)
    summary = company_summary(recs)
    density = correction_density(summary)

    payload = {
        "n_docs": len(recs),
        "n_companies": len(summary),
        "report_distribution": rows,
        "watch_reports": watch,
        "company_summary": summary,
        "correction_density": density[:TOP_CORRECTION],
    }

    if args.report:
        by_corp = find_by_report(recs, args.report)
        print_report_hits(by_corp, args.report)
        payload["query_report"] = {
            "needle": args.report,
            "hits": {name: [{"rcept_dt": r["rcept_dt"], "rcept_no": r["rcept_no"],
                             "report_nm": r["report_nm"],
                             "is_correction": bool(r.get("is_correction"))}
                            for r in rs]
                     for name, rs in by_corp.items()},
        }
    elif args.company:
        rows_t = timeline(recs, args.company, args.dt_from, args.dt_to)
        print_timeline(rows_t, args.company, args.dt_from, args.dt_to)
        payload["query_timeline"] = {
            "company": args.company, "from": args.dt_from, "to": args.dt_to,
            "rows": [{"rcept_dt": r["rcept_dt"], "doc_group": r["doc_group"],
                      "report_nm": r["report_nm"], "rcept_no": r["rcept_no"],
                      "is_correction": bool(r.get("is_correction"))} for r in rows_t],
        }
    else:
        print_overview(rows, watch, summary, density, len(recs))

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print("\nsaved: " + OUT_JSON.name)


if __name__ == "__main__":
    main()
