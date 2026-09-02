# -*- coding: utf-8 -*-
"""표 직독 — 청크의 마크다운 표를 셀 단위로 읽는다.

LLM 추출은 표를 '값: 스칼라'로 납작하게 만들어 라벨·시점·단위를 잃는다.
표는 이미 (행, 열, 단위, 값) 구조이므로 코드가 그대로 읽는다. 연산은
struct_ops가 이 셀들의 구조에서 결정한다 — 질문 표면어를 읽지 않는다.

이 코퍼스의 표는 세 형태다.
1. 서식 표(주요사항보고서): 행 라벨 "N. 항목명(단위)"이 두 칸 반복되고
   하위 라벨(보통주식|기타주식|시작일...)이 오고 값이 반복된다.
2. 부문 표(재무제표 주석): 헤더 행(들)이 열 라벨, "(단위 : X)" 줄이 단위.
   전사|합계|총계|기업 전체 를 포함하는 열은 is_total.
3. 다년도 표(요약재무정보): 열 라벨이 "제N기" 또는 연도.
"""
import re
from collections import namedtuple
from decimal import Decimal, InvalidOperation

# period: 정렬 가능한 문자열 키 (YYYYMMDD 또는 YYYY 또는 "제N기"의 N을 4자리로)
Cell = namedtuple("Cell", "doc_id doc_dt period table_id row_label col_label "
                          "unit value is_total")

_UNIT_DECL_RE = re.compile(r"\(\s*단위\s*[:：]\s*([^)\s,]+)")
_LABEL_UNIT_RE = re.compile(r"\((원|주|천원|백만원|억원|조원|%|퍼센트|명|건)\)\s*$")
_FORM_NO_RE = re.compile(r"^\s*(?:-\s*)?\d+(?:-\d+)?\.\s*")
_NUM_RE = re.compile(r"^[△(\-]?\s*[\d,]+(?:\.\d+)?\s*\)?%?$")
_DATE_CELL_RE = re.compile(r"\d{4}\s*년|\d{4}[-./]\d{1,2}")
_NTH_RE = re.compile(r"제\s?(\d+)\s?기")
_YEAR_RE = re.compile(r"(20\d\d)")
_TOTAL_RE = re.compile(r"전사|합계|총계|전\s?체")
# 당기/전기류 표 구분 마커 — base_year 대비 오프셋
_PERIOD_MARKS = (("당반기", 0), ("전반기", -1), ("당분기", 0), ("전분기", -1),
                 ("당기", 0), ("전기", -1), ("전전기", -2))

KNOWN_UNITS = {"원", "천원", "백만원", "억원", "조원", "주", "%", "퍼센트", "명", "건"}


def _num(cell):
    """셀 전체가 숫자면 Decimal, 아니면 None. △·(x)·-x는 음수."""
    s = cell.strip()
    if not s or not _NUM_RE.match(s):
        return None
    neg = s.startswith(("△", "(", "-"))
    s = s.strip("△()-%").replace(",", "").strip()
    if not s:
        return None
    try:
        v = Decimal(s)
    except InvalidOperation:
        return None
    return -v if neg else v


def _cells_of(line):
    s = line.strip()
    if not s.startswith("|") or not s.endswith("|"):
        return None
    return [c.strip() for c in s[1:-1].split("|")]


def _is_sep(cells):
    return all(re.fullmatch(r":?-{2,}:?", c or "---") for c in cells)


def _base_year(report_nm, rcept_dt):
    m = re.search(r"\((20\d\d)\.", report_nm or "")
    if m:
        return int(m.group(1))
    return int((rcept_dt or "2000")[:4]) - 1   # 정기보고서는 전년도 결산


def _strip_label(s):
    s = _FORM_NO_RE.sub("", s or "").strip()
    return _LABEL_UNIT_RE.sub("", s).strip()


def parse_cells(text, doc_id, doc_dt, report_nm=""):
    """청크 텍스트 → list[Cell]. 표가 없으면 빈 리스트."""
    out = []
    base = _base_year(report_nm, doc_dt)
    cur_unit = None
    cur_period = doc_dt or ""
    block = []           # 연속 파이프 행 블록
    tid = 0
    last_heads = {}      # 열 수 → 직전의 완전한 헤더 (퇴화 헤더 블록에 상속)

    def flush():
        nonlocal block, tid
        if block:
            tid += 1
            out.extend(_parse_block(block, doc_id, doc_dt, f"t{tid}",
                                    cur_unit, cur_period, last_heads))
            block = []

    for raw in (text or "").splitlines():
        cells = _cells_of(raw)
        if cells is None:
            flush()
            continue
        if _is_sep(cells):
            continue
        # "| 당기 | (단위 : 백만원) |" 류 — 표 사이의 단위·시점 선언 행
        joined = " ".join(cells)
        mu = _UNIT_DECL_RE.search(joined)
        marked = next((off for mk, off in _PERIOD_MARKS if mk in joined
                       and len(joined) < 40), None)
        if mu or (marked is not None and len(cells) <= 2):
            flush()
            if mu and mu.group(1) in KNOWN_UNITS:
                cur_unit = mu.group(1)
            if marked is not None:
                cur_period = str(base + marked)
            continue
        block.append(cells)
    flush()
    return out


def _parse_block(rows, doc_id, doc_dt, tid, unit_decl, period, last_heads=None):
    """행 블록 하나를 셀로. 서식 표와 행렬 표를 구분한다."""
    # 서식 표: 첫 칸이 "N. 항목명" 이고 첫 두 칸이 같은 행이 절반 이상
    formish = sum(1 for r in rows if len(r) >= 2 and
                  (_FORM_NO_RE.match(r[0]) or r[0] == r[1])) >= max(1, len(rows) // 2)
    if formish:
        return _parse_form(rows, doc_id, doc_dt, tid)
    return _parse_matrix(rows, doc_id, doc_dt, tid, unit_decl, period,
                         last_heads if last_heads is not None else {})


def _parse_form(rows, doc_id, doc_dt, tid):
    """서식 표 — 행 라벨(단위) + 하위 라벨 + 반복 값."""
    out = []
    for r in rows:
        if len(r) < 2:
            continue
        label_raw = r[0]
        row_label = _strip_label(label_raw)
        mu = _LABEL_UNIT_RE.search(label_raw)
        # 하위 라벨: 라벨 반복 뒤 첫 비숫자 칸. 값: 그 뒤 첫 숫자 칸.
        sub, val = "", None
        for c in r[1:]:
            if c == label_raw or not c or c == "-":
                continue
            v = _num(c)
            if v is None:
                if not sub and not _DATE_CELL_RE.search(c) and len(c) <= 30:
                    m2 = _LABEL_UNIT_RE.search(c)
                    if m2 and not mu:
                        mu = m2
                    sub = _strip_label(c)
                continue
            val = v
            break
        if val is None or not row_label:
            continue
        unit = mu.group(1) if mu else None
        out.append(Cell(doc_id, doc_dt, doc_dt or "", f"form:{tid}",
                        row_label, sub or row_label, unit, val, False))
    return out


def _parse_matrix(rows, doc_id, doc_dt, tid, unit_decl, period, last_heads):
    """행렬 표 — 누적 헤더 행 + 데이터 행.

    원문은 표가 페이지·청크 경계에서 재개될 때 최상단 헤더만 반복하고 하단
    헤더(부문명 등)는 반복하지 않는다. 헤더가 퇴화(모든 열 라벨이 동일)한
    블록은 같은 문서에서 직전에 나온 같은 열 수의 완전한 헤더를 상속한다.
    """
    ncol = max(len(r) for r in rows)
    heads = [[] for _ in range(ncol)]
    out = []
    data_seen = False
    for r in rows:
        nums = [(j, _num(c)) for j, c in enumerate(r)]
        numeric = [j for j, v in nums if v is not None]
        if not numeric:
            # 헤더 행 — 열마다 라벨을 쌓는다 (중복은 한 번만)
            if data_seen:
                continue
            for j, c in enumerate(r):
                if c and c not in heads[j]:
                    heads[j].append(c)
            continue
        if not data_seen:   # 첫 데이터 행 — 헤더 확정 시점
            bottoms = {h[-1] for j, h in enumerate(heads) if j > 0 and h}
            parts_b = {b for b in bottoms if not _TOTAL_RE.search(b)}
            # 퇴화: 부분 열을 구분하는 라벨이 하나도 없다(전부 총계류 반복)
            if not parts_b and last_heads.get(ncol):
                heads = [list(h) for h in last_heads[ncol]]
            elif len(parts_b) >= 2:
                last_heads[ncol] = [list(h) for h in heads]
        data_seen = True
        row_label = _strip_label(r[0]) if r and _num(r[0]) is None else ""
        if not row_label:
            continue
        mu = _LABEL_UNIT_RE.search(r[0])
        for j, v in nums:
            if v is None or j == 0:
                continue
            parts = heads[j] if j < ncol else []
            col_label = " ".join(parts) if parts else f"col{j}"
            # 상단 헤더("기업 전체 총계")는 열 전체에 걸쳐 반복된다. total 판정은
            # 최하단 헤더(가장 구체적인 라벨)로만 한다 — "DX 부문"은 부분이다.
            is_total = bool(parts and _TOTAL_RE.search(parts[-1]))
            # 열 라벨의 제N기·연도는 열별 시점이다
            p = period
            mn = _NTH_RE.search(col_label)
            my = _YEAR_RE.search(col_label)
            if mn:
                p = f"{int(mn.group(1)):04d}"
            elif my:
                p = my.group(1)
            unit = (mu.group(1) if mu else None) or unit_decl
            out.append(Cell(doc_id, doc_dt, p, f"m:{tid}", row_label,
                            col_label, unit, v, is_total))
    return out
