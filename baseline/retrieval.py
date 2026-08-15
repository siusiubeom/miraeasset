# -*- coding: utf-8 -*-
"""BM25 lexical retrieval baseline (외부 의존성 없음).

- 토크나이저: 한글 연속열 → 문자 bigram(+전체 토큰), 영숫자 연속열 → 소문자 토큰, 숫자 유지
- 회사 라우팅: universe.csv의 listed/corp 명 + 수동 별칭으로 질문에서 회사 탐지
- 회사별 청크 파일(processed/chunks/<listed>.jsonl)만 로드해 BM25 검색
"""
import csv, json, math, os, re
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "processed" / "chunks"

_WORD = re.compile(r"[가-힣]+|[A-Za-z]+|[0-9][0-9,. ]*")

# 질문 패턴 → 섹션경로/보고서명 패턴 부스트 ("이 질문은 어느 절에 답이 있는가")
SECTION_PRIORS = [
    (r"매출액|매출|영업수익",                r"손익계산서|요약재무정보", 0.7),
    (r"영업이익|당기순이익|순이익|순손실",     r"손익계산서|요약재무정보", 0.7),
    (r"자산총계|부채총계|자본총계|부채비율",   r"재무상태표|요약재무정보", 0.7),
    (r"현금흐름",                          r"현금흐름표", 0.7),
    (r"설비투자|시설투자|[Cc][Aa][Pp][Ee][Xx]", r"유형자산|생산설비|신규시설투자", 0.6),
    (r"배당",                              r"배당", 0.6),
    (r"최대주주|대주주|지분율|보유.?주식",    r"주주에 관한 사항|대량보유|최대주주", 0.5),
    (r"핵심 ?사업|주요 ?사업|사업.{0,6}(변화|내용|개요)", r"사업의 개요|주요 제품", 0.5),
    (r"연구개발|R&D",                       r"연구개발", 0.5),
    (r"임원|보수|직원 ?현황",                r"임원 및 직원", 0.5),
    (r"수주|공급계약",                      r"공급계약", 0.4),
    (r"자기주식|자사주",                    r"자기주식", 0.6),
    (r"유상증자|전환사채|신주인수권|교환사채|\bCB\b|\bBW\b|\bEB\b|자금조달", r"증자|사채", 0.5),
    (r"소송",                              r"소송", 0.6),
    (r"합병|분할|주식교환",                  r"합병|분할|주식교환", 0.5),
    # 기준(연결/별도) 및 보고서 종류 명시 시
    (r"연결\s?(기준|재무|매출|영업|손익)",    r"연결", 0.4),
    (r"별도\s?(기준|재무|매출|영업|손익)",    r"별도", 0.4),
    (r"사업보고서",                         r"사업보고서", 0.4),
    (r"반기보고서",                         r"반기보고서", 0.4),
    (r"분기보고서",                         r"분기보고서", 0.4),
]

SUPERSEDED_PENALTY = 0.55  # 정정으로 대체된 원본 청크의 점수 배율

ALIASES = {  # 통용명 → listed_name (universe 파일명 기준)
    "현대자동차": "현대차", "케이티": "KT", "NC": "엔씨소프트", "엔시소프트": "엔씨소프트",
    "LIG디펜스앤에어로스페이스": "LIG넥스원", "엘아이지넥스원": "LIG넥스원",
    "JYP": "JYP Ent", "제이와이피": "JYP Ent", "포스코홀딩스": "POSCO홀딩스", "포스코": "POSCO홀딩스",
    "네이버": "NAVER", "엘지에너지솔루션": "LG에너지솔루션", "엘지이노텍": "LG이노텍",
    "삼전": "삼성전자", "하이닉스": "SK하이닉스", "LS일렉트릭": "LS ELECTRIC",
    "미래에셋": "미래에셋증권", "에스엠엔터테인먼트": "에스엠", "SM엔터테인먼트": "에스엠",
    "와이지": "와이지엔터테인먼트", "YG": "와이지엔터테인먼트",
}


def tokenize(text: str):
    toks = []
    for m in _WORD.finditer(text.lower()):
        t = m.group().replace(",", "").replace(" ", "")
        if re.match(r"[가-힣]", t):
            toks.append(t)
            toks.extend(t[i:i + 2] for i in range(len(t) - 1))
        else:
            toks.append(t)
    return toks


class Bm25Index:
    def __init__(self, records, k1=1.2, b=0.75):
        self.records = records
        self.k1, self.b = k1, b
        self.doc_tf = []
        self.doc_len = []
        self.df = Counter()
        for r in records:
            tf = Counter(tokenize(r["text"]))
            self.doc_tf.append(tf)
            self.doc_len.append(sum(tf.values()))
            self.df.update(tf.keys())
        self.n = len(records)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 1.0

    def search(self, query: str, topk=10, prior=None):
        q = Counter(tokenize(query))
        scores = [0.0] * self.n
        for term in q:
            df = self.df.get(term)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        if prior is not None:
            for i, s in enumerate(scores):
                if s > 0:
                    scores[i] = s * prior(self.records[i])
        order = sorted(range(self.n), key=lambda i: -scores[i])[:topk]
        return [(self.records[i], scores[i]) for i in order if scores[i] > 0]


class Retriever:
    def __init__(self):
        self.universe = list(csv.DictReader((ROOT / "corpus" / "universe.csv").open(encoding="utf-8-sig")))
        self.name_map = {}  # 별칭 → listed_name
        for r in self.universe:
            listed = r["listed_name"]
            for key in {listed, r["corp_name"], listed.replace(" ", ""), r["corp_name"].replace(" ", "")}:
                self.name_map[key] = listed
        for k, v in ALIASES.items():
            self.name_map[k] = v
        self._cache = OrderedDict()  # LRU: listed_name → Bm25Index
        self._cache_max = int(os.environ.get("INDEX_CACHE_MAX", "4"))
        # 정정 링크: 원본 rcept_no → 이를 대체한 정정본 rcept_no 목록
        self.superseded = {}
        rep_path = ROOT / "processed" / "build_report.json"
        if rep_path.exists():
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            for corr_doc_id, m in rep.get("corr_matches", {}).items():
                corr_rcept = corr_doc_id.rsplit("_", 1)[-1]
                for orig in m.get("matched", []):
                    self.superseded.setdefault(orig["rcept_no"], []).append(corr_rcept)

    def route(self, question: str):
        """질문에 등장하는 회사(1개 이상)를 긴 이름 우선으로 탐지."""
        qn = question.replace(" ", "")
        found = []
        for key in sorted(self.name_map, key=len, reverse=True):
            if len(key) < 2:
                continue
            if key in question or key.replace(" ", "") in qn:
                listed = self.name_map[key]
                if listed not in found and not any(key in f for f in found):
                    found.append(listed)
        # 부분 중복 제거(예: '삼성전자' 탐지 후 '삼성' 계열 오탐 방지: 긴 매치 우선이므로 위에서 처리됨)
        return found

    def index_for(self, listed: str) -> Bm25Index:
        if listed in self._cache:
            self._cache.move_to_end(listed)
            return self._cache[listed]
        path = CHUNK_DIR / f"{listed}.jsonl"
        recs = [json.loads(l) for l in path.open(encoding="utf-8")]
        for r in recs:  # 정정으로 대체된 원본임을 청크에 표시 (답변 생성 시에도 활용)
            r["superseded_by"] = self.superseded.get(r["rcept_no"], [])
        self._cache[listed] = Bm25Index(recs)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return self._cache[listed]

    def make_prior(self, question: str):
        """질문에서 발동된 섹션 사전확률 + 정정 대체본 감점을 곱셈 배율로 반환."""
        active = [(re.compile(sec_pat), w) for q_pat, sec_pat, w in SECTION_PRIORS
                  if re.search(q_pat, question)]
        years = set(re.findall(r"20\d\d", question))
        wants_quarter = bool(re.search(r"분기|반기|Q[1-4]", question))

        def prior(rec):
            mult = 1.0
            target = rec["section_path"] + " " + rec["report_nm"]
            for sec_re, w in active:
                if sec_re.search(target):
                    mult *= 1.0 + w
            # 질문에 연도가 있으면 그 연도의 문서를 우대. 정기공시는 보고서 '기간'
            # (report_nm 예: '사업보고서 (2025.12)')으로만 판정 — 접수연도로 판정하면
            # 2025년에 접수된 FY2024 사업보고서가 잘못 우대된다. 수시·지분공시는 접수연도가 사건연도.
            if years:
                if rec["group"] == "periodic":
                    if any(y in rec["report_nm"] for y in years):
                        mult *= 1.35
                        # 연도만 묻고 분기·반기를 명시하지 않았으면 사업보고서(연간)를 우대
                        if rec["subtype"] == "annual" and not wants_quarter:
                            mult *= 1.2
                elif rec["rcept_dt"][:4] in years or any(y in rec["report_nm"] for y in years):
                    mult *= 1.35
            if rec["superseded_by"]:
                mult *= SUPERSEDED_PENALTY
            return mult

        return prior, active

    RESERVE = 3  # 사전확률 발동 시 해당 섹션 청크에 보장하는 최소 슬롯 수

    def search(self, question: str, topk=10, companies=None):
        comps = companies or self.route(question)
        if not comps:
            return {"companies": [], "hits": [], "priors": []}
        prior, active = self.make_prior(question)

        # 슬롯 보장은 '어느 절에 답이 있는가'를 지목하는 강한 프리어(가중치 0.6↑)에만 적용.
        # 약한 수식 프리어(연결/별도/보고서종류, 0.4)는 점수 배율에만 관여한다.
        strong = [sr for sr, w in active if w >= 0.6] or [sr for sr, _ in active]

        def is_prior_hit(rec):
            target = rec["section_path"] + " " + rec["report_nm"]
            return any(sec_re.search(target) for sec_re in strong)

        per = max(topk // len(comps), 3)
        hits = []
        for c in comps:
            idx = self.index_for(c)
            cands = idx.search(question, topk=max(per * 6, 30), prior=prior)
            main = cands[:per]
            if active:
                # 프리어 섹션 청크가 보장 슬롯만큼 없으면, 후보군의 상위 프리어 청크로
                # 하위 비(非)프리어 청크를 교체 (서술형 요약이 재무제표 표를 밀어내는 문제 방지)
                want = min(self.RESERVE, per)
                have = [h for h in main if is_prior_hit(h[0])]
                extra = [h for h in cands[per:] if is_prior_hit(h[0])]
                while len(have) < want and extra:
                    repl = extra.pop(0)
                    for j in range(len(main) - 1, -1, -1):
                        if not is_prior_hit(main[j][0]):
                            main[j] = repl
                            have.append(repl)
                            break
                    else:
                        break
            hits.extend(main)
        hits.sort(key=lambda x: -x[1])
        return {"companies": comps, "hits": hits[:topk],
                "priors": [sr.pattern for sr, _ in active]}
