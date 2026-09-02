# -*- coding: utf-8 -*-
"""구조 연산 — 셀 구조에서 유효한 연산을 전부 수행한다.

질문에서는 명사(회사·지표·연도·항목)만 본다. 합계·차액·비율 의도는
판정하지 않는다. 어떤 연산이 유효한지는 표 구조가 결정한다:

  diff   같은 (row_label, col_label, unit), 다른 시점  → 후 − 전 (부호 유지)
  sum    같은 (doc, table, 시점, row_label, unit)의 동종 하위 라벨 → 합산
  ratio  같은 (table, 시점, unit, row_label)에 부분 열 + 전체 열 → 부분/전체
  ratio2 전체 열 없음 + 부분들의 합이 같은 문서군 총액과 0.1% 이내 일치
         → 부분합이 곧 전체 (검산 통과 시에만)
  단위가 다른 셀끼리는 어떤 연산도 하지 않는다.
"""
import re
from collections import namedtuple
from decimal import Decimal

Derivation = namedtuple("Derivation", "kind desc value unit docs")

# 동종 하위 라벨 — 서식 표에서 이 집합 안의 라벨들만 합산 대상이다
_HOMOG_GROUPS = ({"보통주식", "기타주식"}, {"보통주식", "우선주식"},
                 {"보통주식", "기타주식", "우선주식"})
_STOPWORDS = {"부문", "합계", "총계", "전체", "기준", "당기", "전기"}

RATIO2_TOL = Decimal("0.001")   # 부분합 = 총액 검산 허용 오차 0.1%


def _fmt(v):
    return f"{v:,.2f}".rstrip("0").rstrip(".") if v % 1 else f"{v:,.0f}"


def _norm(s):
    return re.sub(r"\s+", "", s or "")


def question_entities(question):
    """질문의 명사 신호 — 지표어·항목어·부문어. 동사·의도는 안 본다."""
    q = _norm(question)
    return q


def question_periods(question):
    """질문이 지목한 시점 — 연월('YYYYMM')과 연도('YYYY'). 명사 추출이다."""
    q = question or ""
    yms = [f"{y}{int(m):02d}" for y, m in
           re.findall(r"(20\d\d)\s*년\s*(\d{1,2})\s*월", q)]
    years = [y for y in re.findall(r"(20\d\d)\s*년", q)]
    return yms, sorted(set(years))


def _period_match(period, yms, years):
    p = str(period or "")
    if yms:   # 연월이 지목되면 연월로만 거른다 — 연도로 넓히면 7월 공시가 섞인다
        return any(p.startswith(ym) or (len(p) == 4 and p == ym[:4])
                   for ym in yms)
    if years:
        return p[:4] in years
    return True   # 질문이 시점을 안 지목하면 전부 통과


def _row_relevant(row_label, qn):
    r = _norm(row_label)
    if not r:
        return False
    if r in qn:
        return True
    # "취득예정금액" vs 질문 "취득예정금액을" — 부분 포함이면 위에서 잡힌다.
    # 반대로 질문이 축약형("매출 비중")이면 라벨의 머리 2자 이상 일치를 본다.
    return len(r) >= 3 and r[:2] in qn and r.rstrip("액") in qn


def _col_relevant(col_label, qn):
    for tok in re.split(r"\s+", col_label or ""):
        t = _norm(tok)
        if len(t) >= 2 and t not in _STOPWORDS and t in qn:
            return True
    return False


def _pct(num, den):
    if not den:
        return None
    return (Decimal(num) / Decimal(den) * 100).quantize(Decimal("0.01"))


def _den_score(label):
    """분모 후보 순위 — 최하단이 '전체·전사'인 열이 '부문 합계'보다 바깥이다."""
    s = 0
    if re.search(r"전\s?체|전사", label or ""):
        s += 2
    if "부문 합계" in (label or ""):
        s -= 1
    return s


def derive(cells, question):
    """유효한 연산 전부 → list[Derivation]. 관련 셀이 없으면 빈 리스트."""
    qn = question_entities(question)
    yms, years = question_periods(question)
    rel = [c for c in cells if _row_relevant(c.row_label, qn)
           and _period_match(c.period, yms, years)]
    if not rel:
        return []
    out = []

    # ── sum: 서식 표의 동종 하위 라벨 (문서·표·시점·행·단위별) ────────────
    sums = {}   # (doc_id, table_id, period, row_label, unit) → (합, 셀들)
    groups = {}
    for c in rel:
        if not c.table_id.startswith("form"):
            continue
        groups.setdefault((c.doc_id, c.table_id, c.period, c.row_label, c.unit),
                          []).append(c)
    for key, cs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        labels = {c.col_label for c in cs}
        if len(cs) >= 2 and any(labels <= g for g in _HOMOG_GROUPS):
            total = sum((c.value for c in cs), Decimal(0))
            sums[key] = (total, cs)
            doc, _, period, row, unit = key
            out.append(Derivation(
                "sum",
                f"합계({row}, {period}): "
                + " + ".join(f"{c.col_label} {_fmt(c.value)}" for c in cs)
                + f" = {_fmt(total)}{unit or ''}",
                total, unit, [doc]))

    # ── diff: 같은 (행, 열, 단위), 다른 시점 — 후 − 전, 부호 유지 ─────────
    series = {}
    for c in rel:
        series.setdefault((c.row_label, c.col_label, c.unit), {}) \
              .setdefault(c.period, c)
    for (row, col, unit), by_p in sorted(series.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if len(by_p) < 2:
            continue
        ps = sorted(by_p)
        a, b = by_p[ps[0]], by_p[ps[-1]]
        d = b.value - a.value
        out.append(Derivation(
            "diff",
            f"차감({row}, {col}): {b.period} {_fmt(b.value)} − "
            f"{a.period} {_fmt(a.value)} = {'+' if d >= 0 else '−'}{_fmt(abs(d))}"
            f"{unit or ''}",
            d, unit, sorted({a.doc_id, b.doc_id})))

    # ── diff of sums: 합산 결과끼리도 같은 (행, 단위) 다른 시점이면 차감 ──
    sum_series = {}
    for (doc, _t, period, row, unit), (total, cs) in sums.items():
        sum_series.setdefault((row, unit), {}).setdefault(period, (total, doc))
    for (row, unit), by_p in sorted(sum_series.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if len(by_p) < 2:
            continue
        ps = sorted(by_p)
        (va, da), (vb, db) = by_p[ps[0]], by_p[ps[-1]]
        d = vb - va
        out.append(Derivation(
            "diff",
            f"차감(합계 {row}): {ps[-1]} {_fmt(vb)} − {ps[0]} {_fmt(va)} = "
            f"{'+' if d >= 0 else '−'}{_fmt(abs(d))}{unit or ''}",
            d, unit, sorted({da, db})))

    # ── ratio: 행렬 표에서 부분 열 / 전체 열 ─────────────────────────────
    mat = {}
    for c in rel:
        if c.table_id.startswith("m"):
            mat.setdefault((c.doc_id, c.table_id, c.period, c.row_label, c.unit),
                           []).append(c)
    ratios = {}   # (row, period, 분자 최하단 라벨) → 최고 분모 점수의 파생 하나
    for (doc, _t, period, row, unit), cs in sorted(mat.items(), key=lambda kv: tuple(map(str, kv[0]))):
        totals = [c for c in cs if c.is_total]
        parts = [c for c in cs if not c.is_total]
        if not totals or not parts:
            continue
        # 전체 열이 여럿이면(부문 합계 vs 기업 전체 총계) 가장 바깥 것을 쓴다.
        den = max(totals, key=lambda c: (_den_score(c.col_label), -c.value))
        # 질문이 '전사·기업 전체'를 지목했으면(명사 매칭) 부문 합계는 분모가
        # 아니다 — 부문 합계는 내부거래 제거 전이라 전사와 다르다(L4).
        if re.search(r"전사|기업\s?전체", question or "") \
                and _den_score(den.col_label) < 2:
            continue
        for p in parts:
            if not _col_relevant(p.col_label, qn):
                continue
            pct = _pct(p.value, den.value)
            if pct is None:
                continue
            key = (row, period, p.col_label.split()[-1] if p.col_label else "")
            # 같은 (행, 시점, 부문)이 여러 문서·표에 반복되면 분모가 더 바깥
            # (전체>부문합계)이고 접수일이 늦은(기간을 온전히 덮는) 것을 쓴다.
            rank = (_den_score(den.col_label), p.doc_dt or "")
            prev = ratios.get(key)
            if prev and prev[0] >= rank:
                continue
            ratios[key] = (rank, Derivation(
                "ratio",
                f"비율({row}, {period}): {p.col_label} {_fmt(p.value)} / "
                f"{den.col_label} {_fmt(den.value)} = {pct}%",
                pct, "%", [doc]))
    out.extend(d for _, d in ratios.values())
    # 중복 제거(같은 표가 여러 보고서에 반복 수록된다) 후 상한
    seen, uniq = set(), []
    for d in out:
        if d.desc in seen:
            continue
        seen.add(d.desc)
        uniq.append(d)
    return uniq[:16]


def ratio2(values, cells, row_hint=""):
    """전체 열이 없을 때 — 부분들의 합이 문서군 총액과 일치하면 그 합이 전체.

    values: (라벨, Decimal) 목록 — 서술문에서 추출된 부분 값들.
    cells 의 서식 표 합계(동종 하위 라벨 합) 또는 total 셀들에서 총액 후보를
    만들어, |부분합 − 총액| / 총액 <= 0.1% 인 후보가 있을 때만 비율을 만든다.
    """
    if len(values) < 2:
        return []
    psum = sum((v for _, v in values), Decimal(0))
    # 총액 후보: 문서별 서식 표 합계와 그 문서군 전체 합
    per_doc = {}
    for c in cells:
        if c.table_id.startswith("form") and any(
                {c.col_label} <= g for g in _HOMOG_GROUPS):
            per_doc.setdefault((c.doc_id, c.row_label, c.unit), Decimal(0))
            per_doc[(c.doc_id, c.row_label, c.unit)] += c.value
    # 후보는 문서별 총액 각각과, 같은 (행, 단위)의 문서군 합 — 어느 쪽이든
    # 부분합과 일치하면 그 부분들이 그 총액을 분할하는 구조다.
    candidates = []
    by_row = {}
    for (doc, row, unit), tot in sorted(per_doc.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if row_hint and _norm(row_hint) not in _norm(row):
            continue
        candidates.append(((row, unit), (tot, [doc])))
        s, docs = by_row.get((row, unit), (Decimal(0), []))
        by_row[(row, unit)] = (s + tot, docs + [doc])
    candidates += [(k, v) for k, v in sorted(by_row.items(), key=lambda kv: tuple(map(str, kv[0])))
                   if len(v[1]) > 1]
    out = []
    for (row, unit), (tot, docs) in candidates:
        if not tot or abs(psum - tot) / abs(tot) > RATIO2_TOL:
            continue
        for label, v in values:
            pct = _pct(v, psum)
            out.append(Derivation(
                "ratio2",
                f"비율({row}): {label} {_fmt(v)} / 부분합 {_fmt(psum)} = {pct}% "
                f"(부분합이 문서 총액 {_fmt(tot)}{unit or ''}과 "
                f"{(abs(psum - tot) / tot * 100):.4f}% 오차로 일치 — 전체로 인정)",
                pct, "%", sorted(docs)))
        break   # 가장 먼저 검산을 통과한 총액 기준 하나만
    return out
