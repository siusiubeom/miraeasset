# -*- coding: utf-8 -*-
"""지표어 → 절(section_path) 매핑 — 검색이 필요 없는 질문을 검색에서 뺀다.

정기보고서는 법정 서식이라 70개사가 같은 목차를 쓴다. "매출액이 어디에 있는가"는
찾을 문제가 아니라 아는 문제다. 지금은 make_prior가 이걸 가중치로만 처리해
BM25 점수에 묻힌다(Q01: 요약재무정보에 사전확률을 줬는데도 경영진단 서술문이
이겼다).

절 패턴만으로는 수시공시를 놓친다. "자기주식"은 정기보고서의 '주식의 총수 등'
절에도 있고 주요사항보고서(자기주식취득결정) 문서 자체이기도 하다. 그래서 각
항목은 절 패턴과 보고서명 패턴을 함께 갖고, 둘 중 하나만 맞아도 후보로 둔다.
"""
import re

# 항목 순서 = 우선순위. 구체적인 것을 앞에 둔다("배당"이 "매출액"보다 먼저).
# question: 질문에서 이 항목으로 보내는 지표어
# sections: section_path 패턴
# reports : report_nm 패턴 (수시·지분공시를 놓치지 않기 위한 통로)
SECTION_MAP = (
    {
        "name": "배당",
        "question": r"배당",
        "sections": (r"배당에\s*관한\s*사항", r"배당금", r"이익잉여금"),
        "reports": (r"현금[·ㆍ]?현물배당", r"배당"),
    },
    {
        # 자본금·액면은 주석 항목이다. '발행주식'이 자기주식 항목으로 먼저
        # 빨려 들어가지 않도록 앞에 둔다.
        "name": "자본금",
        "question": r"자본금|액면|납입자본|주식발행초과금",
        "sections": (r"자본금", r"주식의\s*총수"),
        "reports": (),
    },
    {
        "name": "우발·충당",
        "question": r"우발|충당부채|채무보증|지급보증",
        "sections": (r"우발", r"충당"),
        "reports": (r"채무보증", r"담보제공"),
    },
    {
        "name": "특수관계자",
        "question": r"특수관계자|계열사\s?거래|내부거래",
        "sections": (r"특수관계자", r"계열회사"),
        "reports": (),
    },
    {
        "name": "무형자산·손상",
        "question": r"개발비|무형자산|영업권|손상",
        "sections": (r"무형자산", r"영업권", r"손상"),
        "reports": (),
    },
    {
        "name": "리스",
        "question": r"리스|사용권자산",
        "sections": (r"리스", r"사용권자산"),
        "reports": (),
    },
    {
        "name": "파생·위험관리",
        "question": r"파생|헤지|위험회피",
        "sections": (r"파생", r"위험관리", r"금융위험"),
        "reports": (r"파생상품",),
    },
    {
        "name": "자기주식",
        "question": r"자기주식|자사주|발행주식|유통주식",
        "sections": (r"주식의\s*총수", r"자기주식"),
        "reports": (r"자기주식", r"주식의\s*총수"),
    },
    {
        "name": "주주·지분",
        "question": r"최대주주|지분율|주주|대주주|보유\s?주식",
        "sections": (r"주주에\s*관한\s*사항", r"최대주주"),
        "reports": (r"대량보유", r"최대주주", r"임원[·ㆍ]?주요주주"),
    },
    {
        "name": "임원 보수",
        "question": r"임원\s?보수|보수\s?총액|보수|급여",
        "sections": (r"임원의\s*보수", r"임원\s*및\s*직원"),
        "reports": (r"임원의\s*보수",),
    },
    {
        "name": "차입금·사채",
        "question": r"차입금|사채|유동성|자금조달|회사채",
        "sections": (r"유동성\s*및\s*자금조달", r"증권의\s*발행을\s*통한\s*자금조달"),
        "reports": (r"사채", r"증자", r"자금조달"),
    },
    {
        "name": "부문",
        "question": r"부문|세그먼트|사업부",
        "sections": (r"부문", r"재무상태\s*및\s*영업실적", r"기타\s*참고사항"),
        "reports": (),
    },
    {
        "name": "계약·수주",
        "question": r"계약|수주|공급",
        "sections": (r"공시내용\s*진행\s*및\s*변경사항",),
        "reports": (r"단일판매", r"공급계약", r"수주"),
    },
    {
        "name": "손익지표",
        "question": r"매출액|매출|영업이익|영업손실|당기순이익|순이익|영업수익",
        # r"재무제표"까지 넣으면 '연결재무제표 주석'(회계정책 주석)이 후보에
        # 섞여 손익 표를 밀어낸다(Q03이 1위→6위로 밀렸다). 표 절만 남긴다.
        "sections": (r"요약재무정보", r"손익계산서"),
        "reports": (),
    },
)

# 절 지정은 정기보고서의 정형 목차를 전제로 한다. 이 그룹의 문서는 절 패턴이
# 맞지 않아도 보고서명 패턴으로 후보에 들어올 수 있다.
EVENT_GROUPS = ("major", "exchange", "holding")


def _compile(entry):
    return ([re.compile(p) for p in entry["sections"]],
            [re.compile(p) for p in entry["reports"]])


_COMPILED = [(e, *_compile(e)) for e in SECTION_MAP]


def resolve_sections(question):
    """질문 → (매칭 판정 함수용 패턴 묶음, 매핑된 지표어). 해당 없으면 (None, None).

    반환하는 패턴 묶음은 (절 패턴 리스트, 보고서명 패턴 리스트, 항목 이름)이다.
    """
    q = question or ""
    for entry, sec_res, rep_res in _COMPILED:
        m = re.search(entry["question"], q)
        if m:
            return (sec_res, rep_res, entry["name"]), m.group(0)
    return None, None


def matches(rec, patterns):
    """청크가 지정 절 후보에 드는가."""
    sec_res, rep_res, _ = patterns
    path = rec.get("section_path") or ""
    name = rec.get("report_nm") or ""
    if any(r.search(path) for r in sec_res):
        return True
    return bool(rep_res) and any(r.search(name) for r in rep_res)
