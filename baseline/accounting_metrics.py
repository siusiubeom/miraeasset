"""회계 좌표가 일치하는 매출액과 영업이익으로 영업이익률을 계산한다."""
from dataclasses import dataclass
from decimal import Decimal
import re

import struct_ops as SO


@dataclass(frozen=True)
class Fact:
    entity: str
    basis: str
    period: str
    duration: str
    metric: str
    segment: str
    unit: str
    value: Decimal
    source: str
    table: str


def operating_margin(profit: Fact, revenue: Fact):
    """금액 비중과 영업이익률의 분모를 구분. 불명확·다른 좌표는 계산하지 않는다."""
    coordinates = ("entity", "basis", "period", "duration", "segment", "unit", "source", "table")
    if any(not getattr(profit, k) or getattr(profit, k) == "unknown" for k in coordinates):
        return None
    if any(getattr(profit, k) != getattr(revenue, k) for k in coordinates):
        return None
    if profit.metric != "영업이익" or revenue.metric != "매출액":
        return None
    if profit.unit not in ("원", "천원", "백만원", "억원", "조원") or revenue.value <= 0:
        return None
    return (profit.value / revenue.value * 100).quantize(Decimal("0.01"))


def _basis(rec):
    path = re.sub(r"\s+", "", rec.get("section_path", ""))
    if "연결" in path:
        return "연결"
    if "별도" in path or re.search(r">4-[1-5]\.", path):
        return "별도"
    return "unknown"


def derive_margins(cells, question, hits):
    """질문이 지목한 지표·부문에 대해 동일 표의 영업이익/매출액을 산출한다."""
    q = SO.question_entities(question)
    if not (("매출" in q and "영업이익" in q) or "영업이익률" in q):
        return []
    yms, years = SO.question_periods(question)
    metas = {(r["rcept_no"], r.get("section_path", "")): r for r, _ in hits}
    groups, total_flags = {}, {}
    for c in cells:
        if c.row_label not in ("매출액", "영업이익") or not SO._period_match(c.period, yms, years):
            continue
        if not c.table_id.startswith("m:"):
            continue
        # 합계 열도 '기업 전체'임이 명시된 것만 사용. 부분합/내부거래조정은 배제.
        if c.is_total:
            if SO._den_score(c.col_label) < 2:
                continue
        elif not SO._col_relevant(c.col_label, q):
            continue
        section = c.table_id.split(":", 2)[2] if c.table_id.count(":") >= 2 else ""
        rec = metas.get((c.doc_id, section))
        if not rec:
            continue
        # 연도만 지정한 경우 연간 보고서의 연간 값에 한정한다.
        if "사업보고서" not in rec.get("report_nm", ""):
            continue
        f = Fact(rec.get("corp", ""), _basis(rec), c.period, "annual", c.row_label,
                 c.col_label, c.unit or "unknown", c.value, c.doc_id, c.table_id)
        key = (f.entity, f.basis, f.period, f.segment, f.unit, f.source, f.table)
        groups.setdefault(key, {}).setdefault(f.metric, []).append(f)
        total_flags[key] = c.is_total
    out, margins = [], []
    for key, metrics in groups.items():
        if set(metrics) != {"매출액", "영업이익"}:
            continue
        # 같은 좌표의 상충 값을 검색 순서로 고르지 않는다.
        if any(len({f.value for f in fs}) != 1 for fs in metrics.values()):
            continue
        profit, revenue = metrics["영업이익"][0], metrics["매출액"][0]
        pct = operating_margin(profit, revenue)
        if pct is None:
            continue
        margins.append((profit, pct, total_flags[key]))
        out.append(SO.Derivation("margin",
            f"영업이익률({profit.entity}, {profit.basis}, {profit.period}, {profit.segment}): "
            f"영업이익 {SO._fmt(profit.value)}{profit.unit} / 매출액 "
            f"{SO._fmt(revenue.value)}{revenue.unit} × 100 = {pct}%",
            pct, "%", [profit.source]))
    # 같은 표의 부문/전사 이익률을 비교한다. 사업상 원인은 추정하지 않는다.
    for part, part_pct, is_total in margins:
        if is_total:
            continue
        for whole, whole_pct, whole_is_total in margins:
            coordinates = ("entity", "basis", "period", "duration", "unit", "source", "table")
            if not whole_is_total or whole.value == 0 or any(getattr(part, k) != getattr(whole, k) for k in coordinates):
                continue
            relation = "보다 높다" if part_pct > whole_pct else "보다 낮다" if part_pct < whole_pct else "와 표시 정밀도에서 같다"
            out.append(SO.Derivation("margin_relation",
                f"이익률 비교({part.entity}, {part.basis}, {part.period}): "
                f"{part.segment}의 영업이익률 {part_pct}%는 전사 {whole_pct}%{relation}. "
                "두 비중은 각각 매출액과 영업이익을 기준으로 계산하며, "
                "부문과 전사의 매출액 대비 영업이익 비율이 다르면 두 비중도 달라진다.",
                part_pct, "%", [part.source]))
    return out
