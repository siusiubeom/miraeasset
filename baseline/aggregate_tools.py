# -*- coding: utf-8 -*-
"""manifest로 문서를 필터하고 본문에서 값과 사유를 집계한다. 조회 건수와 추출 성공 건수를 구분한다."""
import gzip
import json
import os
import re
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.jsonl"
CHUNK_DIR = ROOT / "processed" / "chunks"

# 값 셀 : "| 계약금액(원) | 314,500,000,000 | 314,500,000,000 |"
_FIELD_VALUE_RE = "{field}\\s*\\(?[^)|\\n]*\\)?\\s*\\|\\s*([\\d,]+(?:\\.\\d+)?)"
# 사유 셀 : "| 3. 정정사유 | 계약금액, 종료일 변경 | ..."
_REASON_RE = "{field}\\s*\\|\\s*([^|\\n]{{2,60}})"

_manifest_cache = None
_chunk_cache = OrderedDict()
_CHUNK_CACHE_MAX = 4


def load_manifest():
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")
                           if l.strip()]
    return _manifest_cache


def chunks_by_rcept(corp):
    """회사의 청크를 접수번호별로 묶어 돌려준다(전수)."""
    if corp in _chunk_cache:
        _chunk_cache.move_to_end(corp)
        return _chunk_cache[corp]
    path = CHUNK_DIR / f"{corp}.jsonl"
    out = {}
    opener = path.open if path.exists() else None
    if opener is None:
        path = CHUNK_DIR / f"{corp}.jsonl.gz"
        opener = lambda **kw: gzip.open(path, "rt", **kw)
    # 검색과 구조 연산이 같은 배포 산출물을 읽는다. 파일 부재를 빈 원장으로 숨기지 않는다.
    with opener(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out.setdefault(r["rcept_no"], []).append(r)
    _chunk_cache[corp] = out
    while len(_chunk_cache) > _CHUNK_CACHE_MAX:
        _chunk_cache.popitem(last=False)
    return out


def docs_for(corp, filters=None):
    """원장에서 조건에 맞는 문서를 전수 수집한다. 검색이 아니다."""
    f = filters or {}
    out = []
    for m in load_manifest():
        if m["listed_name"] != corp and m["corp_name"] != corp:
            continue
        if f.get("report") and not re.search(f["report"], m["report_nm"]):
            continue
        if f.get("year") and not m["rcept_dt"].startswith(str(f["year"])):
            continue
        if f.get("group") and m["doc_group"] != f["group"]:
            continue
        if f.get("correction") is not None and bool(m["is_correction"]) != f["correction"]:
            continue
        out.append(m)
    return sorted(out, key=lambda m: (m["rcept_dt"], m["rcept_no"]))


def _doc_text(corp, rcept_no):
    return "\n".join(c.get("text") or "" for c in chunks_by_rcept(corp).get(rcept_no, []))


_NUM_CELL_RE = re.compile(r"\|\s*([\d,]+(?:\.\d+)?)\s*(?=\|)")


def parse_field(text, field):
    """문서 본문에서 숫자 필드를 뽑는다. 정정본은 '정정후' 열이 뒤에 온다.

    값은 셀 전체가 숫자인 칸에서만 취한다. 서식 표는 행 라벨이 두 번 반복되므로
    라벨 접두("2. 취득예정금액")의 행번호가 값으로 잡히면 안 된다(T01: 최대값 2원).
    """
    vals = []
    for line in (text or "").splitlines():
        if field not in line:
            continue
        for m in _NUM_CELL_RE.finditer(line):
            try:
                vals.append(float(m.group(1).replace(",", "")))
            except ValueError:
                continue
    return vals


# "2021-03-11일자 매매기준환율인 1USD = 1,140.70원을 적용하여"
_FX_RE = re.compile(
    r"(?:(\d{4}-\d{2}-\d{2})\s*일?자\s*)?매매기준환율[^\d]{0,12}"
    r"1\s*(USD|달러|EUR|JPY|유로|엔)?\s*=?\s*([\d,]+(?:\.\d+)?)\s*원")


def parse_fx(text):
    """계약금액 원화 환산에 적용된 환율과 기준일. 없으면 None.

    환율은 계약 건마다 다르다. 컨텍스트에 여러 문서를 함께 실으면 모델이 다른
    건의 환율을 가져온다(E05: 채택 문서의 1,140.70 대신 1,325.70을 썼다).
    """
    m = _FX_RE.search(text or "")
    if not m:
        return None
    return {"basis_date": m.group(1) or "", "currency": m.group(2) or "USD",
            "rate": m.group(3)}


def extremum(corp, filters=None, field="계약금액", mode="max", superseded=None):
    """조건에 맞는 문서 전수에서 field의 최대/최소 문서를 찾는다.

    정정으로 대체된 원본은 제외한다 : 폐기된 값이 순위를 뒤집을 수 있다.
    """
    superseded = superseded or {}
    docs, rows = docs_for(corp, filters), []
    for m in docs:
        if superseded.get(m["rcept_no"]):
            continue                      # 정정 대체된 원본은 최종본이 따로 있다
        text = _doc_text(corp, m["rcept_no"])
        vals = parse_field(text, field)
        if vals:
            # 같은 표에 정정전·정정후가 나란히 오면 뒤쪽(정정후)을 취한다.
            rows.append({"rcept_no": m["rcept_no"], "rcept_dt": m["rcept_dt"],
                         "report_nm": m["report_nm"], "value": vals[-1],
                         "n_values": len(vals), "fx": parse_fx(text)})
    if not rows:
        return {"n_docs": len(docs), "n_parsed": 0, "picked": None, "rows": []}
    rows.sort(key=lambda r: r["value"], reverse=(mode == "max"))
    tied = [r for r in rows if r["value"] == rows[0]["value"]]
    return {"n_docs": len(docs), "n_parsed": len(rows), "picked": rows[0],
            "tied": tied if len(tied) > 1 else [], "field": field, "mode": mode,
            "rows": rows[:10]}


def sort_by_date(corp, filters=None):
    """조건에 맞는 문서 전체를 접수일순으로. 개수 제한 없음."""
    docs = docs_for(corp, filters)
    return {"n_docs": len(docs),
            "rows": [{"rcept_dt": m["rcept_dt"], "rcept_no": m["rcept_no"],
                      "report_nm": m["report_nm"], "group": m["doc_group"],
                      "is_correction": bool(m["is_correction"])} for m in docs]}


def correction_changes(text):
    """정정 전후 표의 바뀐 필드만 읽는다. 사유 문장을 임의로 분해하지 않는다."""
    out, seen = [], set()
    columns = None
    for line in (text or "").splitlines():
        if not line.strip().startswith("|"):
            columns = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        compact = [re.sub(r"\s+", "", c) for c in cells]
        if all(label in compact for label in ("정정항목", "정정전", "정정후")):
            columns = tuple(compact.index(label) for label in ("정정항목", "정정전", "정정후"))
            continue
        if not columns or max(columns) >= len(cells):
            continue
        field, before, after = (cells[i] for i in columns)
        if not field or re.fullmatch(r"[-:]+", field) or before == after:
            continue
        normalized = re.sub(r"\s+", "", field)
        # 공시 서식의 필드명. 계약기간의 시작/종료일 변경은 같은 사유 범주다.
        category = ("계약금액" if "계약금액" in normalized else
                    "계약기간" if "계약기간" in normalized else None)
        key = (field, before, after)
        if key not in seen:
            seen.add(key)
            out.append(dict(field=field, category=category, before=before, after=after))
    return out


def distinct_count(corp, filters=None, field="정정사유"):
    """조건에 맞는 문서 전수에서 사유를 뽑아 유형별로 센다.

    동수면 임의로 하나를 고르지 않는다 : 그 사실 자체가 답이다.
    """
    docs = docs_for(corp, filters)
    counter, per_doc, missing = Counter(), [], 0
    occurrences, changed_docs = Counter(), 0
    for m in docs:
        text = _doc_text(corp, m["rcept_no"])
        changes = correction_changes(text) if os.environ.get("USE_CORRECTION_OCCURRENCES", "1") != "0" else []
        if changes:
            changed_docs += 1
            occurrences.update({c["category"] for c in changes if c["category"]})
        found = re.findall(_REASON_RE.format(field=re.escape(field)), text)
        seen = None
        for raw in found:
            v = re.sub(r"\s+", " ", raw).strip(" |")
            if v and v != field and not v.startswith(field):
                seen = v
                break
        if seen:
            counter[seen] += 1
            per_doc.append({"rcept_no": m["rcept_no"], "rcept_dt": m["rcept_dt"],
                            "reason": seen, "changes": changes})
        else:
            missing += 1
            if changes:
                per_doc.append({"rcept_no": m["rcept_no"], "rcept_dt": m["rcept_dt"],
                                "reason": "정정사유 필드 미추출", "changes": changes})
    if not counter:
        return {"n_docs": len(docs), "n_parsed": 0, "counts": {}, "top": [],
                "tied": False, "per_doc": per_doc, "missing": missing,
                "change_occurrences": dict(occurrences), "n_change_docs": changed_docs,
                "n_unresolved_change_docs": len(docs) - changed_docs}
    top_n = max(counter.values())
    top = sorted(k for k, v in counter.items() if v == top_n)
    return {"n_docs": len(docs), "n_parsed": sum(counter.values()),
            "counts": dict(counter.most_common()), "top": top, "top_n": top_n,
            "tied": len(top) > 1, "per_doc": per_doc, "missing": missing,
            "change_occurrences": dict(occurrences), "n_change_docs": changed_docs,
            "n_unresolved_change_docs": len(docs) - changed_docs}
