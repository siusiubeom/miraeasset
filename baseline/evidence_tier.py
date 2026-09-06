# -*- coding: utf-8 -*-
"""문서 유형과 절 경로로 검색 가중치 등급을 분류한다. 판정이 불명확하면 추정 등급을 반환한다."""
import re

# 사다리 각 단계의 판정 표현
# 마크다운 표 구분선. 청킹에서 표는 "|---|---|" 형태로 남는다.
TABLE_SEP_RE = re.compile(r"\|\s*-{3,}")

# ① 법정 서식 : 주요사항보고서·거래소공시. 정정 이력이 추적되는 서식만.
STATUTORY_GROUPS = ("major", "exchange")

# ② 감사받은 재무제표 본문. section_path의 공백을 걷어낸 뒤 대조한다
#    (원문이 "(첨부)연 결 재 무 제 표"처럼 자간을 벌려 쓴다).
AUDITED_FS_RE = re.compile(
    r"감사보고서.*?(?:첨부)?(?:연결)?재무제표"
    r"|재무상태표|손익계산서|포괄손익계산서|현금흐름표|자본변동표")

# ③ 주석 : 감사받은 재무제표의 일부이자 감사의견의 대상이다.
#    절 이름에 '주석'을 달고 오는 것(3. 연결재무제표 주석)과, 번호 항목으로
#    풀려 들어오는 것(III. 재무에 관한 사항 > 18. 자본금, 37. 부문정보 (연결))이
#    둘 다 주석이다. 후자를 놓치면 tier 3이 사실상 비어 버린다.
NOTES_RE = re.compile(r"주석")
NOTE_ITEM_RE = re.compile(r"재무에관한사항>\d[\d\-.]*\.")
# 번호 항목이지만 주석이 아닌 것 : 재무제표 본문과 요약표
NOT_A_NOTE_RE = re.compile(r"요약재무정보|(?<!주석)(?:연결)?재무제표$")

# ④ 정기보고서
PERIODIC_GROUP = "periodic"

# ⑤ 서술 : 표가 없고 조·억 단위 어림 표기가 있는 문장
NARRATIVE_UNIT_RE = re.compile(r"\d+\s*(?:조|억)\s*원|\d+\s*조\b|\d+\s*억\b")

# ⑥ 인용·요약 : 다른 문서를 가리키는 표현
CITATION_RE = re.compile(
    r"참조하시기\s?바랍|참고하시기\s?바랍|기재를?\s?생략|"
    r"상기\s?(?:내용|사항|공시)|별도로?\s?공시|해당\s?공시를?\s?참조|"
    r"요약한\s?것|자세한\s?(?:사항|내용)은")

DEFAULT_TIER = 4        # 사다리 어디에도 안 걸릴 때
TIERS = (1, 2, 3, 4, 5, 6)


def _norm_path(section_path):
    """자간을 벌려 쓴 절 경로를 대조 가능한 형태로."""
    return re.sub(r"\s+", "", section_path or "")


def classify(rec):
    """청크 → (tier, tier_confident, 판정근거).

    tier_confident=False는 추정 등급이라는 뜻이다. 추정 등급으로 순위를
    흔들지 않기 위해 호출부는 가중치를 적용하지 않는다.
    """
    group = rec.get("group") or ""
    path = _norm_path(rec.get("section_path"))
    text = rec.get("text") or ""
    has_table = bool(TABLE_SEP_RE.search(text))

    # ① 법정 서식의 정형 필드 표. 조건을 느슨하게 하면 등급이 위로 밀려
    #    위계가 무의미해지므로, 정정 이력이 추적되는 서식 + 표에만 준다.
    if group in STATUTORY_GROUPS and has_table:
        return 1, True, f"법정 서식({group})의 정형 표"

    # ② 외부감사인의 감사를 받은 재무제표 본문 표
    if AUDITED_FS_RE.search(path) and has_table and not NOTES_RE.search(path):
        return 2, True, "감사받은 재무제표 본문 표"

    # ③ 감사받은 재무제표의 주석
    if NOTES_RE.search(path):
        return 3, True, "재무제표 주석"
    if NOTE_ITEM_RE.search(path) and not NOT_A_NOTE_RE.search(path):
        return 3, True, "재무제표 주석(번호 항목)"

    # ④ 정기보고서의 정형 표
    if group == PERIODIC_GROUP and has_table:
        return 4, True, "정기보고서의 정형 표"

    # ⑤ 경영진이 직접 서술한 문장 : 표가 없고 조·억 어림 표기가 있다
    if not has_table and NARRATIVE_UNIT_RE.search(text):
        return 5, True, "서술형 문장(조·억 단위 어림 표기)"

    # ⑥ 다른 문서를 인용·요약한 것
    if CITATION_RE.search(text):
        return 6, True, "다른 문서 인용·요약"

    # ⑦ 어디에도 해당 없음 : 추정 등급. 절 경로가 비어 있으면 특히 그렇다.
    reason = "절 경로 없음" if not path else "사다리 미해당"
    return DEFAULT_TIER, False, f"{reason} : 추정 등급 {DEFAULT_TIER}"


def annotate(rec):
    """청크에 evidence_tier / tier_confident / tier_reason 을 붙여 돌려준다."""
    tier, confident, reason = classify(rec)
    rec["evidence_tier"] = tier
    rec["tier_confident"] = confident
    rec["tier_reason"] = reason
    return rec


def tier_label(tier):
    return {
        1: "법정 서식 표", 2: "감사받은 재무제표 표", 3: "재무제표 주석",
        4: "정기보고서 표", 5: "서술형 문장", 6: "인용·요약",
    }.get(tier, f"tier {tier}")
