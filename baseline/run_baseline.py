# -*- coding: utf-8 -*-
"""검색 베이스라인 평가.

질문 유형은 과제요약의 예시 6유형(검색/비교/복합추론 × Closed/Open)을 본떠 구성.
채점(검색 단계만):
  - route_ok      : 회사 라우팅 정답 여부
  - doc_hit@k     : top-k 청크 중 기대 문서(rcept_no 집합)에서 나온 것이 있는가
  - evidence_hit@k: top-k 청크 텍스트에 기대 근거(정규식)가 있는가
"""
import json, re, sys, time
from pathlib import Path
from retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
TOPK = 10

MANI = [json.loads(l) for l in (ROOT / "corpus" / "manifest.jsonl").open(encoding="utf-8")]


def docs(corp_name=None, group=None, subtype=None, base_year=None, rcept_prefix=None, nm_re=None):
    out = set()
    for m in MANI:
        if corp_name and m["corp_name"] != corp_name: continue
        if group and m["doc_group"] != group: continue
        if subtype and m["doc_subtype"] != subtype: continue
        if base_year and m.get("base_year") != base_year: continue
        if rcept_prefix and not m["rcept_no"].startswith(rcept_prefix): continue
        if nm_re and not re.search(nm_re, m["report_nm"]): continue
        out.add(m["rcept_no"])
    return out


QUESTIONS = [
    # ── 검색·정보추출 (Closed)
    dict(qid="Q01", type="추출/Closed", q="삼성전자의 2025년 연결기준 매출액은 얼마인가?",
         companies=["삼성전자"], gold_docs=docs("삼성전자", "periodic", "annual", 2025),
         evidence=r"333,605,938"),
    dict(qid="Q02", type="추출/Closed", q="LG에너지솔루션의 2025년 연결기준 영업이익은 얼마인가?",
         companies=["LG에너지솔루션"], gold_docs=docs("LG에너지솔루션", "periodic", "annual", 2025),
         evidence=r"영업이익"),
    dict(qid="Q03", type="추출/Closed(별칭)", q="현대차의 2025년 연결기준 매출액은 얼마인가?",
         companies=["현대차"], gold_docs=docs("현대자동차", "periodic", "annual", 2025),
         evidence=r"매출액|영업수익"),
    # ── 검색·정보추출 (Open)
    dict(qid="Q04", type="추출/Open", q="시프트업의 2024년 사업보고서를 기준으로 주요 사업 내용을 정리해줘",
         companies=["시프트업"], gold_docs=docs("시프트업", "periodic", "annual", 2024),
         evidence=r"사업의 개요|주요 제품"),
    # ── 다중조회·비교 (Closed)
    dict(qid="Q05", type="비교/Closed", q="2차전지 기업 삼성SDI와 에코프로비엠 중 2025년 매출액이 더 큰 기업은 어디인가?",
         companies=["삼성SDI", "에코프로비엠"],
         gold_docs=docs("삼성SDI", "periodic", "annual", 2025) | docs("에코프로비엠", "periodic", "annual", 2025),
         evidence=r"매출액", need_both=True),
    # ── 다중조회·정리 (Open)
    dict(qid="Q06", type="정리/Open", q="셀트리온이 2025년에 결정한 자기주식 취득 내역을 정리해줘",
         companies=["셀트리온"], gold_docs=docs("셀트리온", "major", rcept_prefix="2025", nm_re="자기주식취득"),
         evidence=r"자기주식"),
    # ── 복합 추론 / 정정 연결
    dict(qid="Q07", type="정정연결", q="삼성전자가 2025년 7월 체결한 반도체 위탁생산 공급계약의 계약상대는 어디인가?",
         companies=["삼성전자"], gold_docs={"20250731800028"},  # 정정본(테슬라 공개)
         evidence=r"테슬라|Tesla"),
    dict(qid="Q08", type="복합/이벤트", q="삼성전자가 2024년 11월에 결정한 자기주식 취득 규모는 얼마인가?",
         companies=["삼성전자"], gold_docs=docs("삼성전자", "major", rcept_prefix="202411"),
         evidence=r"자기주식"),
    dict(qid="Q09", type="복합/지분", q="알테오젠의 최대주주는 누구인가?",
         companies=["알테오젠"], gold_docs=docs("알테오젠", "holding"),
         evidence=r"박순재"),
    dict(qid="Q10", type="복합/계약", q="레인보우로보틱스의 최대주주 변경을 수반하는 콜옵션 주주간계약의 콜옵션 권리자는 누구인가?",
         companies=["레인보우로보틱스"], gold_docs=docs("레인보우로보틱스", "exchange"),
         evidence=r"삼성전자"),
    # ── 수주 공시
    dict(qid="Q11", type="수주/Open", q="한화오션이 2025년에 공시한 단일판매·공급계약 체결 내역을 정리해줘",
         companies=["한화오션"], gold_docs=docs("한화오션", "exchange", rcept_prefix="2025", nm_re="공급계약체결"),
         evidence=r"계약금액"),
    # ── 정보한계 (코퍼스에 없어야 정상)
    dict(qid="Q12", type="정보한계", q="삼성전자의 2026년 3분기 실적 전망은 어떻게 되는가?",
         companies=["삼성전자"], gold_docs=set(), evidence=None, unanswerable=True),
]


def main():
    ret = Retriever()
    results = []
    for spec in QUESTIONS:
        t0 = time.time()
        routed = ret.route(spec["q"])
        res = ret.search(spec["q"], topk=TOPK)
        dt = time.time() - t0
        hits = res["hits"]
        hit_docs = [h[0]["rcept_no"] for h in hits]
        hit_corps = {h[0]["corp"] for h in hits}
        # 정정본이 원본보다 위에 랭크되는지 (둘 다 top-k에 있을 때만 판정).
        # 정정 체인이 로딩되지 않았으면 superseded_by가 전부 비어 무조건 통과처럼
        # 보이므로, 통과가 아니라 '판정 불가'로 구분해 리포트한다.
        corr_order_ok = True if ret.chain_loaded else None
        rank_of = {}
        for i, h in enumerate(hits):
            rank_of.setdefault(h[0]["rcept_no"], i)
        for i, h in enumerate(hits):
            for corr in h[0].get("superseded_by", []):
                if corr in rank_of and rank_of[corr] > i:
                    corr_order_ok = False
        route_ok = set(spec["companies"]) <= set(routed)
        doc_hit = bool(spec["gold_docs"] & set(hit_docs))
        doc_rank = next((i + 1 for i, d in enumerate(hit_docs) if d in spec["gold_docs"]), None)
        ev_hit = bool(spec["evidence"] and any(re.search(spec["evidence"], h[0]["text"]) for h in hits))
        both_ok = (not spec.get("need_both")) or set(spec["companies"]) <= hit_corps
        results.append({
            "qid": spec["qid"], "type": spec["type"], "q": spec["q"],
            "routed": routed, "route_ok": route_ok,
            "doc_hit": doc_hit, "doc_rank": doc_rank, "evidence_hit": ev_hit,
            "both_companies_in_topk": both_ok,
            "corr_above_orig": corr_order_ok,
            "active_priors": res.get("priors", []),
            "unanswerable": spec.get("unanswerable", False),
            "latency_sec": round(dt, 2),
            "top3": [{"chunk_id": h[0]["chunk_id"], "score": round(h[1], 2),
                      "section": h[0]["section_path"][:90]} for h in hits[:3]],
        })
        corr_s = "판정불가" if corr_order_ok is None else corr_order_ok
        print(f"{spec['qid']} route={route_ok} doc_hit={doc_hit} rank={doc_rank} "
              f"ev={ev_hit} corr={corr_s} {dt:.1f}s", flush=True)

    scored = [r for r in results if not r["unanswerable"]]
    summary = {
        "topk": TOPK,
        "n_questions": len(scored),
        "route_acc": sum(r["route_ok"] for r in scored) / len(scored),
        "doc_hit_at_k": sum(r["doc_hit"] for r in scored) / len(scored),
        "evidence_hit_at_k": sum(r["evidence_hit"] for r in scored) / len(scored),
        "mean_latency_sec": round(sum(r["latency_sec"] for r in results) / len(results), 2),
        "max_latency_sec": max(r["latency_sec"] for r in results),
        # 정정 체인 미로딩 시 corr_above_orig는 전부 None(판정 불가)이며,
        # 통과율로 집계하면 100%처럼 보이므로 별도 필드로 상태를 남긴다.
        "corr_chain_loaded": ret.chain_loaded,
        "corr_above_orig_acc": (
            None if not ret.chain_loaded
            else sum(bool(r["corr_above_orig"]) for r in scored) / len(scored)),
    }
    out = {"summary": summary, "results": results}
    # 기본 파일명은 공유 파일이므로, 인자로 결과 파일명을 지정해 분리할 수 있게 한다.
    #   python run_baseline.py results_seongmin_baseline.json
    out_name = sys.argv[1] if len(sys.argv) > 1 else "baseline_results.json"
    (Path(__file__).parent / out_name).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved: {out_name}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
