# -*- coding: utf-8 -*-
"""전문가 문항 10개 end-to-end 실행 — 검색이 아니라 답변을 본다.

run_adversarial.py와 같은 방식으로 answer_question을 그대로 태우되, 문항마다
검색·위계 장치가 실제로 어떻게 동작했는지를 함께 기록한다.

  python baseline/run_expert.py                 전체 실행 → expert10.json
  python baseline/run_expert.py out.json E01 E03  일부만

기록 항목
  - section_route 발동 여부와 폴백 여부
  - top-8 청크의 evidence_tier 분포
  - tier_demoted(위계 가중치로 밀려난 하위 tier 청크)
  - 호출 수, 소요 시간, TPM(input + maxTokens)
  - 답변의 모든 수치가 retrieved_context에 실재하는지
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import answerer as A

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHUNK_DIR = ROOT / "processed" / "chunks"

QUESTIONS = [
    ("E01", "삼성전자",
     "삼성전자의 2024년 11월 자기주식 취득 결정 공시의 자기주식 취득금액 한도 산식에서 "
     "6번 항목 '직전 사업연도말 이후 자기주식 처분시 처분주식의 취득원가'만 더하기로 "
     "계산된다. 나머지는 모두 빼기다. 6번이 더하기인 이유를 설명하라."),
    ("E02", "삼성전자",
     "삼성전자의 2023년 별도기준 손익계산서에서 영업손실이 11조 5,263억원인데 당기순이익은 "
     "25조 3,971억원이다. 같은 해 연결기준 당기순이익 15조 4,871억원보다 별도 순이익이 크다. "
     "두 현상의 원인을 손익계산서 항목별로 설명하라."),
    ("E03", "삼성전자",
     "삼성전자의 2025년 사업보고서 자본금 주석에 발행주식 액면총액 673,561백만원과 "
     "납입자본금 897,514백만원이 다르다고 기재되어 있다. 차액의 원인과 그 차액이 어느 수치와 "
     "일치하는지 검산하라."),
    ("E04", "삼성전자",
     "삼성전자의 2024년 11월 자기주식 취득 결정에서 1일 매수 주문수량 한도가 보통주 "
     "6,525,123주, 우선주 691,203주로 산정됐다. 산출근거 세 항목으로 도출 과정을 보이고, "
     "보통주와 우선주에서 다른 항목이 채택된 이유를 설명하라."),
    ("E05", "한화오션",
     "한화오션이 2023년에 공시한 단일판매·공급계약 중 계약금액이 가장 큰 건을 찾고, "
     "원화 환산에 적용된 환율과 기준일을 밝혀라."),
    ("E06", "삼성E&A",
     "삼성E&A가 2024년에 접수한 단일판매·공급계약 정정 건의 정정 사유를 유형별로 집계하고 "
     "가장 빈번한 사유를 답하라."),
    ("E07", "현대모비스",
     "현대모비스의 최대주주의 최대주주의 최대주주는 누구인가."),
    ("E08", "에코프로비엠",
     "에코프로비엠이 발행한 전환사채 중 실제로 주식으로 전환된 물량과 전환 시점을 알려달라."),
    ("E09", "OCI홀딩스",
     "OCI홀딩스의 2023년과 2025년 매출액을 각각 제시하고 증감률을 계산하라."),
    ("E10", "에스엠",
     "에스엠의 2023년 최대주주 변동 과정을 시간순으로 재구성하고, 각 시점의 보고자와 "
     "서식상 최대주주를 구분해 밝혀라."),
]

# 답변 수치 대조에서 제외 — 접수번호(14자리)와 접수일(8자리)은 출처 표기다.
_NUM_RE = re.compile(r"\d[\d,]{3,}")


def has_chunks(company):
    return (CHUNK_DIR / f"{company}.jsonl").exists()


def ungrounded_numbers(answer, context):
    """답변 수치 중 컨텍스트에 없는 것. 코드 계산 결과도 여기서는 그대로 잡는다."""
    ctx = {re.sub(r"[^0-9]", "", t) for t in re.findall(r"\d[\d,]*", context or "")}
    out = []
    for tok in _NUM_RE.findall(answer or ""):
        d = re.sub(r"[^0-9]", "", tok.strip(","))
        if "," not in tok and len(d) in (8, 14):
            continue
        if d and d not in ctx and tok not in out:
            out.append(tok)
    return out


def route_summary(notes):
    """section_route 기록 → (발동 여부, 절 이름, 폴백 여부)."""
    fired = [n for n in notes if n.startswith("[2+]")]
    fell = [n for n in notes if n.startswith("[2!]")]
    section = ""
    if fired:
        m = re.search(r"→\s*([^(]+)", fired[0])
        section = m.group(1).strip() if m else ""
    return bool(fired), section, bool(fell)


def run_one(qid, company, question):
    t0 = time.time()
    A.begin_request()
    d = A.answer_question(qid, question)
    stats = A.call_stats()
    req = A._REQ
    notes = list(req.get("section_route") or [])
    fired, section, fell = route_summary(notes)
    demoted = [{"chunk_id": r.get("chunk_id"),
                "tier": r.get("evidence_tier"),
                "section": (r.get("section_path") or r.get("report_nm") or "").split(" > ")[-1]}
               for r in (req.get("tier_demoted") or [])]
    bad = ungrounded_numbers(d["answer"], d["retrieved_context"])
    return {
        "qid": qid, "company": company, "question": question,
        "answer": d["answer"], "think_trace": d["think_trace"],
        "retrieved_context_chars": len(d["retrieved_context"]),
        "section_route": {"fired": fired, "section": section,
                          "fell_back": fell, "notes": notes},
        "tier_hist": dict(sorted(Counter(
            t for t in (req.get("hit_tiers") or []) if t is not None).items())),
        "tier_demoted": demoted,
        "calls": stats["n_calls"], "tpm_cost": stats["tpm_cost"],
        "sec": round(time.time() - t0, 2),
        "ungrounded_numbers": bad,
    }


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "expert10.json"
    only = set(sys.argv[2:])

    rows, skipped = [], []
    for qid, company, question in QUESTIONS:
        if only and qid not in only:
            continue
        if not has_chunks(company):
            skipped.append({"qid": qid, "company": company,
                            "reason": f"전사 청크 없음: processed/chunks/{company}.jsonl"})
            print(f"[{qid}] SKIP — {company} 전사 청크 없음")
            continue
        r = run_one(qid, company, question)
        rows.append(r)
        route = (f"{r['section_route']['section']}"
                 + ("+폴백" if r["section_route"]["fell_back"] else "")) \
            if r["section_route"]["fired"] else ("폴백만" if r["section_route"]["fell_back"] else "미발동")
        print(f"[{qid}] {r['company']} {r['sec']}s calls={r['calls']} tpm={r['tpm_cost']} "
              f"route={route} tier={r['tier_hist']} 강등={len(r['tier_demoted'])} "
              f"미근거수치={len(r['ungrounded_numbers'])}")
        print("    " + r["answer"].replace("\n", " ")[:180])

    payload = {
        "n": len(rows), "skipped": skipped,
        "aggregate": {
            "calls_mean": round(sum(r["calls"] for r in rows) / len(rows), 2) if rows else 0,
            "sec_mean": round(sum(r["sec"] for r in rows) / len(rows), 2) if rows else 0,
            "tpm_max": max((r["tpm_cost"] for r in rows), default=0),
            "route_fired": sum(r["section_route"]["fired"] for r in rows),
            "route_fellback": sum(r["section_route"]["fell_back"] for r in rows),
            "with_ungrounded": sum(1 for r in rows if r["ungrounded_numbers"]),
        },
        "results": rows,
    }
    (HERE / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    print("\n집계:", json.dumps(payload["aggregate"], ensure_ascii=False))
    if skipped:
        print("건너뜀:", ", ".join(f"{s['qid']}({s['company']})" for s in skipped))
    print("saved:", out_name)


if __name__ == "__main__":
    main()
