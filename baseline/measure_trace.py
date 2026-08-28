# -*- coding: utf-8 -*-
"""think_trace 품질 계량.

눈으로 "산문이 됐다", "만점급이다"를 판단해 온 것을 숫자로 바꾼다. 개정 전후를
같은 지표로 재야 개선인지 우연인지 구분된다. 표준 라이브러리만 사용한다.

실행: python baseline/measure_trace.py <결과JSON> <라벨>
입력 JSON: [{"qid"|"question_id", "question", "think_trace", "answer"}, ...]
출력: 문항별 표 + 집계 + trace_metrics_<라벨>.json
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 기준선 (Küster 2024, 유럽 상장사 KAM 실측) ────────────────────────────────
BASE_BOILERPLATE = 55.0     # 정형구 비율 %, 목표 30 이하
BASE_NUM_DENSITY = 2.0      # 수치 밀도 %, 목표 5 이상
TARGET_BOILERPLATE = 30.0
TARGET_NUM_DENSITY = 5.0
MIN_IDENTIFIERS = 3         # 문항당 고유 식별자 최소 개수
MAX_SENTENCES = 6           # [판단] 길이 상한
NGRAM_N = 5                 # 정형구 판정 n-gram (어절)

LOG_SEP = "---\n[시스템 로그]"

# 시스템이 강제하는 고정 문구 — 정형구 집계에서 분리한다
FIXED_PHRASES = (
    "미래 실적 전망이나 투자 의견",
    "추출값과 계산값이 불일치해",
    "개인의 생년월일·주소·연락처",
    "대상 70개사에 포함되지",
    "공시 수집 범위",
    "조회된 공시에 해당 항목이",
)

RCEPT_RE = re.compile(r"\b\d{14}\b")
DATE_RE = re.compile(r"\b20\d\d[-.]\d{1,2}[-.]\d{1,2}\b|"
                     r"20\d\d\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?")
REPORT_RE = re.compile(r"(?:\[기재정정\])?[가-힣A-Za-z·ㆍ]*보고서(?:\s*\([\d.]+\))?|"
                       r"주요사항보고서\([^)]+\)|단일판매[^\s,)]*")
SECTION_RE = re.compile(r"[^\s]+\s*>\s*[^\s]+")
NUM_TOKEN_RE = re.compile(r"\d")
SENT_SPLIT_RE = re.compile(r"(?<=[.?!다요])\s+")

# 미래지향·평가 표현 (answerer._OPINION_RE 와 금지어)
FORWARD_RE = re.compile(
    r"전망|예상|예측|오를|내릴|떨어질|좋아질|나빠질|괜찮을|"
    r"투자\s?의견|매수|매도|목표\s?주가|추천|"
    r"모멘텀|유망|우려|저평가|고평가|매력적|공격적|아마도|추정컨대|것으로\s?보인다")
# 기각 서술
REJECT_RE = re.compile(r"제외했|채택하지\s?않았|쓸\s?수\s?없|폐기|전환했|기각|대신 |걸린 것이 없")
# 결과 서술 (절차 나열과 판단 기록을 가르는 지표)
RESULT_RE = re.compile(r"한\s?결과|확인한\s?결과|조회하면|나오지만|판정했|취했|판단했|밝혔")


def split_trace(t: str):
    """모델 산문 부분과 [시스템 로그] 부분을 분리."""
    if LOG_SEP in t:
        head, _, tail = t.partition(LOG_SEP)
        return head.strip(), tail.strip()
    # 구분자 없이 코드 로그만 남은 경우 — 모델 산문은 0으로 본다
    if t.lstrip().startswith("[") and "회사 라우팅" in t:
        return "", t.strip()
    return t.strip(), ""


def words(t):
    return [w for w in re.split(r"\s+", t) if w]


def ngrams(ws, n=NGRAM_N):
    return {" ".join(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}


def strip_fixed(t: str) -> str:
    """시스템 고정 문구가 포함된 문장을 제거한 텍스트."""
    keep = [s for s in SENT_SPLIT_RE.split(t)
            if not any(f in s for f in FIXED_PHRASES)]
    return " ".join(keep)


def measure_one(row):
    prose, log = split_trace(row.get("think_trace", "") or "")
    ans = row.get("answer", "") or ""
    ws = words(prose)
    sents = [s for s in SENT_SPLIT_RE.split(prose) if s.strip()]

    identifiers = set(RCEPT_RE.findall(prose)) | set(DATE_RE.findall(prose)) \
        | set(m.group(0).strip() for m in REPORT_RE.finditer(prose) if len(m.group(0)) > 3) \
        | set(m.group(0) for m in SECTION_RE.finditer(prose))

    num_tokens = sum(1 for w in ws if NUM_TOKEN_RE.search(w))
    rejects = REJECT_RE.findall(prose)
    # 기각 서술의 진위: trace 산문이 든 접수번호가 시스템 로그에 실재하는가
    log_rcepts = set(RCEPT_RE.findall(log))
    fabricated = [no for no in RCEPT_RE.findall(prose)
                  if rejects and no not in log_rcepts] if rejects else []

    # 소제목(v1.3 §2): 첫 줄이 짧은 한 구절이고 문장으로 끝나지 않아야 한다.
    first = prose.splitlines()[0].strip() if prose.strip() else ""
    has_subtitle = bool(first) and len(first) <= 24 and not first.endswith((".", "다", "요"))

    return {
        "qid": row.get("qid") or row.get("question_id"),
        "subtitle": first[:24] if has_subtitle else (sents[0][:24] if sents else ""),
        "has_subtitle": has_subtitle,
        "sentences": len(sents),
        "words": len(ws),
        "num_density": round(100 * num_tokens / len(ws), 1) if ws else 0.0,
        "identifiers": len(identifiers),
        "identifier_list": sorted(identifiers)[:5],
        "has_rejection": bool(rejects),
        "fabricated_rejection": fabricated,
        "result_ratio": round(100 * sum(1 for s in sents if RESULT_RE.search(s))
                              / len(sents), 1) if sents else 0.0,
        # 거절 문구 자체("예측·투자 의견은 제공하지 않습니다")는 미래지향 주장이
        # 아니라 그 반대이므로 고정 문구를 걷어낸 뒤 센다.
        "forward_hits": (len(FORWARD_RE.findall(strip_fixed(prose)))
                         + len(FORWARD_RE.findall(strip_fixed(ans)))),
        "over_length": len(sents) > MAX_SENTENCES,
        "_prose": prose,
    }


def boilerplate(rows, strip=False):
    """문항 쌍마다 5-gram 중복률을 계산해 평균."""
    texts = [strip_fixed(r["_prose"]) if strip else r["_prose"] for r in rows]
    grams = [ngrams(words(t)) for t in texts]
    pairs = [(a, b) for a, b in combinations(range(len(rows)), 2) if grams[a] and grams[b]]
    if not pairs:
        return 0.0
    vals = [100 * len(grams[a] & grams[b]) / min(len(grams[a]), len(grams[b]))
            for a, b in pairs]
    return round(sum(vals) / len(vals), 1)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "baseline/adversarial_results.json")
    label = sys.argv[2] if len(sys.argv) > 2 else "run"
    rows = [measure_one(r) for r in json.loads(src.read_text(encoding="utf-8"))]

    print(f"{'Q':<4}{'소제목':<26}{'문장':>4}{'수치%':>7}{'식별자':>6}"
          f"{'기각':>5}{'결과%':>7}{'미래':>5}")
    for r in rows:
        print(f"{r['qid']:<4}{r['subtitle'][:24]:<26}{r['sentences']:>4}"
              f"{r['num_density']:>7}{r['identifiers']:>6}"
              f"{'O' if r['has_rejection'] else '-':>5}{r['result_ratio']:>7}"
              f"{r['forward_hits']:>5}")

    n = len(rows)
    bp_all = boilerplate(rows)
    bp_free = boilerplate(rows, strip=True)
    fabricated = [r for r in rows if r["fabricated_rejection"]]
    agg = {
        "label": label, "n": n,
        "boilerplate_pct": bp_all,
        "boilerplate_pct_excl_fixed": bp_free,
        "num_density_pct": round(sum(r["num_density"] for r in rows) / n, 1),
        "identifiers_mean": round(sum(r["identifiers"] for r in rows) / n, 1),
        "identifiers_below_min": sum(1 for r in rows if r["identifiers"] < MIN_IDENTIFIERS),
        "rejection_rate_pct": round(100 * sum(r["has_rejection"] for r in rows) / n, 1),
        "fabricated_rejections": len(fabricated),
        "result_ratio_pct": round(sum(r["result_ratio"] for r in rows) / n, 1),
        "forward_hits_total": sum(r["forward_hits"] for r in rows),
        "sentences_mean": round(sum(r["sentences"] for r in rows) / n, 1),
        "over_length_count": sum(r["over_length"] for r in rows),
        "subtitle_rate_pct": round(100 * sum(r["has_subtitle"] for r in rows) / n, 1),
    }

    def mark(ok):
        return "통과" if ok else "미달"

    print("\n[집계]")
    print(f"  정형구           {bp_all}%  (고정문구 제외 {bp_free}%) "
          f"기준선 {BASE_BOILERPLATE}% 목표 {TARGET_BOILERPLATE}% 이하 "
          f"— {mark(bp_free <= TARGET_BOILERPLATE)}")
    print(f"  수치 밀도        {agg['num_density_pct']}%  기준선 {BASE_NUM_DENSITY}% "
          f"목표 {TARGET_NUM_DENSITY}% 이상 — {mark(agg['num_density_pct'] >= TARGET_NUM_DENSITY)}")
    print(f"  고유 식별자      평균 {agg['identifiers_mean']}개, "
          f"{MIN_IDENTIFIERS}개 미만 {agg['identifiers_below_min']}문항 "
          f"— {mark(agg['identifiers_below_min'] == 0)}")
    print(f"  미래지향 표현    {agg['forward_hits_total']}회  목표 0 "
          f"— {mark(agg['forward_hits_total'] == 0)}")
    print(f"  기각 서술        {agg['rejection_rate_pct']}%  목표 50% 이상 "
          f"— {mark(agg['rejection_rate_pct'] >= 50)}")
    print(f"  지어낸 기각      {agg['fabricated_rejections']}건  목표 0 "
          f"— {mark(agg['fabricated_rejections'] == 0)}")
    if fabricated:
        for r in fabricated:
            print(f"      {r['qid']}: {r['fabricated_rejection']}")
    print(f"  소제목           {agg['subtitle_rate_pct']}%  목표 100% "
          f"— {mark(agg['subtitle_rate_pct'] == 100)}")
    print(f"  결과 서술 비율   {agg['result_ratio_pct']}%")
    print(f"  길이             평균 {agg['sentences_mean']}문장, "
          f"{MAX_SENTENCES}문장 초과 {agg['over_length_count']}문항")

    for r in rows:
        r.pop("_prose", None)
    out = Path(__file__).parent / f"trace_metrics_{label}.json"
    out.write_text(json.dumps({"aggregate": agg, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved: {out.name}")


if __name__ == "__main__":
    main()
