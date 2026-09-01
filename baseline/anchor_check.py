# -*- coding: utf-8 -*-
"""문장 단위 접지 검사 — "근거에 닿지 않는 문장은 내보내지 않는다".

금지어 목록은 완결되지 않는다. 늘릴수록 오탐이 늘고(_OPINION_RE가 '1일 매수
주문수량 한도'를 잡았다), 목록에 없는 새 표현은 그대로 나간다. 추측 표현은
증상이고 원인은 근거 없는 주장이다. 근거를 강제하면 표현은 저절로 사라진다.

SYSTEM_PROMPT §5("원본 정보를 생성하지 않는다")를 프롬프트 규칙이 아니라
구조로 강제하는 층이다.
"""
import re

MIN_LEN = 30          # 이보다 짧은 문장은 연결·판정 문장으로 보고 통과시킨다
UNANCHORED_TRACE_RATIO = 0.5   # trace 미접지 비율이 이를 넘으면 별도 표기

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=다)\s*\n|(?<=습니다)\s+|(?<=입니다)\s+")
_RCEPT_RE = re.compile(r"\b\d{14}\b")
_BIGNUM_RE = re.compile(r"\d[\d,]{3,}")
_DATE_RES = (
    re.compile(r"20\d\d[-.]\d{1,2}[-.]\d{1,2}"),
    re.compile(r"20\d\d\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?"),
    re.compile(r"\b20\d{6}\b"),
)
# 마크다운 표의 첫 열 — 서식 항목명이 여기 온다(취득예정금액, 처분목적 등)
_TABLE_ROW_RE = re.compile(r"^\s*\|([^|\n]{2,40})\|", re.M)
_HANGUL_TERM_RE = re.compile(r"[가-힣][가-힣0-9\s·ㆍ()]{3,}")
# 숫자 비교는 쉼표를 걷어내고 한다
_COMMA_RE = re.compile(r",")

MIN_TERM_LEN = 4      # 표 항목명·절 마디는 4자 이상만 앵커로 쓴다


def _digits(tok):
    return re.sub(r"[^0-9]", "", tok or "")


def build_anchors(context, question="", calc_values=(), hits=(), extra_terms=()):
    """컨텍스트·질문·코드 계산 결과에서 접지 후보를 모은다.

    반환은 문자열 집합이다. 문장에 이 중 하나라도 들어 있으면 접지된 것으로 본다.
    수치는 원문 표기와 쉼표를 뗀 표기를 함께 넣는다.
    """
    ctx = context or ""
    anchors = set()

    def add_num(tok):
        t = tok.strip(",")
        if len(_digits(t)) >= 4:
            anchors.add(t)
            anchors.add(_digits(t))

    for tok in _BIGNUM_RE.findall(ctx):
        add_num(tok)
    anchors |= set(_RCEPT_RE.findall(ctx))
    for rx in _DATE_RES:
        anchors |= set(rx.findall(ctx))

    # 표 첫 열의 한글 항목명
    for cell in _TABLE_ROW_RE.findall(ctx):
        c = cell.strip()
        if len(c) >= MIN_TERM_LEN and _HANGUL_TERM_RE.fullmatch(c):
            anchors.add(c)

    for rec, _ in hits or ():
        nm = rec.get("report_nm") or ""
        if nm:
            anchors.add(nm)
            anchors.update(p.strip() for p in re.split(r"[()\[\]]", nm)
                           if len(p.strip()) >= MIN_TERM_LEN)
        for part in (rec.get("section_path") or "").split(">"):
            p = re.sub(r"^\s*[IVX0-9.\-]+\s*", "", part).strip()
            if len(p) >= MIN_TERM_LEN:
                anchors.add(p)
        for key in ("corp_name", "corp"):
            if rec.get(key):
                anchors.add(rec[key])

    # 질문에 제시된 수치·용어도 접지로 인정한다. 질문이 준 값을 되받아 검산하는
    # 답변(E03)을 미접지로 판정하면 정답이 지워진다.
    for tok in _BIGNUM_RE.findall(question or ""):
        add_num(tok)
    for rx in _DATE_RES:
        anchors |= set(rx.findall(question or ""))

    # 코드 계산 결과(합·차·비율·환산)는 컨텍스트에 없지만 근거 있는 값이다.
    for v in calc_values or ():
        add_num(str(v))
        anchors.add("{:,}".format(int(v)) if str(v).lstrip("-").isdigit() else str(v))

    anchors |= {t for t in (extra_terms or ()) if len(t) >= MIN_TERM_LEN}
    return {a for a in anchors if a}


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


def is_anchored(sentence, anchors):
    plain = _COMMA_RE.sub("", sentence)
    for a in anchors:
        if a in sentence or (a.isdigit() and a in plain):
            return True
    return False


def unanchored_sentences(text, anchors, min_len=MIN_LEN):
    """min_len 이상인데 앵커를 하나도 포함하지 않는 문장들."""
    return [s for s in split_sentences(text)
            if len(s) >= min_len and not is_anchored(s, anchors)]


# 삭제 후 남는 번호·불릿 정리 — "2. 3."만 남은 목록을 내보내지 않기 위해
_LIST_MARK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def renumber(text):
    """문장 제거 뒤 남은 목록의 번호를 다시 매기고 빈 항목을 지운다."""
    out, n = [], 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        m = _LIST_MARK_RE.match(stripped)
        if m:
            body = stripped[m.end():].strip()
            if not body:
                continue          # 내용이 사라진 항목은 버린다
            if m.group(0).strip()[0].isdigit():
                n += 1
                out.append(f"{n}. {body}")
            else:
                out.append(f"- {body}")
        else:
            if stripped:
                out.append(line.rstrip())
            elif out and out[-1] != "":
                out.append("")
    return "\n".join(out).strip()


def has_direct_answer(text, min_len=12):
    """제거 후에도 직답으로 볼 만한 문장이 남았는가."""
    for s in split_sentences(text):
        body = _LIST_MARK_RE.sub("", s)
        if len(body) >= min_len:
            return True
    return False
