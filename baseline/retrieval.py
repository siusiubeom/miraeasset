# -*- coding: utf-8 -*-
"""BM25 lexical retrieval baseline (외부 의존성 없음).

- 토크나이저: 한글 연속열 → 문자 bigram(+전체 토큰), 영숫자 연속열 → 소문자 토큰, 숫자 유지
- 회사 라우팅: universe.csv의 listed/corp 명 + 수동 별칭으로 질문에서 회사 탐지
- 회사별 청크 파일(processed/chunks/<listed>.jsonl)만 로드해 BM25 검색
"""
import csv, gzip, json, math, os, re, sys
from collections import Counter, OrderedDict
from pathlib import Path

from evidence_tier import annotate
from section_map import matches as section_matches, resolve_sections

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
    # 주석에만 있는 항목들 — 목적적합성 판정을 프리어가 맡는다
    (r"자본금|액면|납입자본|주식발행초과금",   r"자본금", 0.6),
    (r"우발|충당부채|채무보증|지급보증",      r"우발|충당", 0.6),
    (r"특수관계자|계열사\s?거래",            r"특수관계자", 0.6),
    (r"부문|세그먼트",                      r"부문", 0.6),
    (r"개발비|무형자산|영업권|손상",          r"무형자산|영업권|손상", 0.6),
    (r"리스|사용권자산",                    r"리스", 0.6),
    (r"파생|헤지|위험회피",                 r"파생|위험관리", 0.6),
    (r"합병|분할|주식교환",                  r"합병|분할|주식교환", 0.5),
    # 기준(연결/별도) 및 보고서 종류 명시 시
    (r"연결\s?(기준|재무|매출|영업|손익)",    r"연결", 0.4),
    (r"별도\s?(기준|재무|매출|영업|손익)",    r"별도", 0.4),
    (r"사업보고서",                         r"사업보고서", 0.4),
    (r"반기보고서",                         r"반기보고서", 0.4),
    (r"분기보고서",                         r"분기보고서", 0.4),
]

SUPERSEDED_PENALTY = 0.55  # 정정으로 대체된 원본 청크의 점수 배율

# ── 증거 위계 가중치 (감사기준서 500) ────────────────────────────────────────
# 정정 감점이 위계의 시점 축이라면 이쪽은 문서 유형 축이다. BM25는 지표어가
# 조밀한 서술문을 표보다 위로 올리는데, 정확한 값은 표에만 있다.
# K-IFRS에서 주석(tier 3)은 재무제표의 일부이자 감사의견의 대상이다. 반면
# 요약재무정보(tier 4)는 감사받은 수치를 사업보고서에 옮겨 적은 것이라 감사
# 대상이 아니다. 신뢰성 축에서 3은 2와 4 사이에 온다.
# 목적적합성(질문이 그 절을 요구하는가)은 이 가중치가 아니라 SECTION_PRIORS와
# 지정형 경로가 맡는다 — 감사기준 500이 적합성을 두 축으로 가르는 그대로다.
_DEFAULT_TIER_WEIGHT = {1: 1.5, 2: 1.4, 3: 1.35, 4: 1.2, 5: 0.7, 6: 0.6}


def _load_tier_weight():
    """EVIDENCE_TIER_WEIGHT="1:1.5,5:0.6" 형식으로 덮어쓸 수 있다."""
    w = dict(_DEFAULT_TIER_WEIGHT)
    raw = os.environ.get("EVIDENCE_TIER_WEIGHT", "").strip()
    for part in raw.split(","):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        try:
            w[int(k.strip())] = float(v)
        except ValueError:
            print(f"[warn] EVIDENCE_TIER_WEIGHT 항목을 읽지 못함: {part!r}", file=sys.stderr)
    return w


TIER_WEIGHT = _load_tier_weight()


def use_evidence_tier() -> bool:
    """기본 켜짐. USE_EVIDENCE_TIER=0 으로 끈다(전후 비교용 스위치)."""
    return os.environ.get("USE_EVIDENCE_TIER", "1") not in ("0", "false", "False")


# 수치를 묻는 질문에만 가중치를 건다. 계약상대·목적·대상자처럼 서술에 답이
# 있는 질문에서 표를 위로 올리면 오히려 정답을 밀어낸다.
# 금액·수량을 다루는 질문. 계약상대·목적·대상자처럼 서술에 답이 있는 질문에는
# 가중치를 걸지 않는다. '액면총액·자본금'처럼 "얼마"라는 말 없이 수치를 묻는
# 표현이 있어 좁게 넓혔다(Q13은 이것이 없어 가중치가 발동하지 않았다).
NUMERIC_QUESTION_RE = re.compile(
    r"얼마|금액|매출|영업이익|순이익|비중|비율|몇|규모|"
    r"총액|자본금|액면|주식수|주식\s?총수|단가|수량")


# 경영진단·부문 서술 절에만 있는 정확한 수치가 있다(부문별 실적, 기재 비중).
# 이 유형의 질문에서는 서술형(tier 5) 강등을 완화한다 — 강등하면 답이 사라진다.
NARRATIVE_OK_QUESTION_RE = re.compile(r"부문|세그먼트|비중|사업부")
TIER5_RELAXED = 0.9


def use_section_route() -> bool:
    """지정형 경로. 기본 켜짐, USE_SECTION_ROUTE=0으로 끈다.

    BM25를 버리는 게 아니라 검색이 필요 없는 질문을 검색에서 빼는 것이다.
    지정 절 안에서의 순위는 여전히 BM25가 정한다.
    """
    return os.environ.get("USE_SECTION_ROUTE", "1") not in ("0", "false", "False")

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
        # build_report.json이 없으면 링크가 통째로 비고 정정 감점·감지가 조용히 죽는다.
        # 무증상으로 지나가지 않도록 chain_loaded 플래그로 상류에 전파한다.
        self.superseded = {}
        self.supersedes = {}   # 정정본 rcept_no → 그것이 정정한 원본 rcept_no 목록
        self.chain_loaded = False
        rep_path = ROOT / "processed" / "build_report.json"
        if rep_path.exists():
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            for corr_doc_id, m in rep.get("corr_matches", {}).items():
                corr_rcept = corr_doc_id.rsplit("_", 1)[-1]
                for orig in m.get("matched", []):
                    self.superseded.setdefault(orig["rcept_no"], []).append(corr_rcept)
                    # 역방향: 정정본 → 그것이 정정한 대상. 감점으로 원본이 top-k
                    # 밖으로 밀려나도 '체인 말단을 취했다'는 판단 근거는 남아야 한다.
                    self.supersedes.setdefault(corr_rcept, []).append(orig["rcept_no"])
            self.chain_loaded = bool(self.superseded)
            if not self.chain_loaded:
                print(f"[warn] {rep_path} 의 corr_matches에 매칭된 정정 체인이 0건 — "
                      "정정 감점·정정본 우선 판정이 비활성 상태입니다.", file=sys.stderr)
        else:
            print(f"[warn] {rep_path} 없음 — 정정 체인 미로딩. 정정 감점"
                  f"(SUPERSEDED_PENALTY={SUPERSEDED_PENALTY})과 정정 대체 원본 감지가 "
                  "동작하지 않습니다. preprocess/build_company_md.py 를 실행해 생성하십시오.",
                  file=sys.stderr)

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
        # 배포 환경에서는 청크를 gzip으로 싣는다(1.5GB → 약 240MB). 둘 다 지원.
        path = CHUNK_DIR / f"{listed}.jsonl"
        if path.exists():
            fh = path.open(encoding="utf-8")
        else:
            fh = gzip.open(CHUNK_DIR / f"{listed}.jsonl.gz", "rt", encoding="utf-8")
        with fh:
            recs = [json.loads(l) for l in fh if l.strip()]
        for r in recs:  # 정정으로 대체된 원본임을 청크에 표시 (답변 생성 시에도 활용)
            r["superseded_by"] = self.superseded.get(r["rcept_no"], [])
            r["supersedes"] = self.supersedes.get(r["rcept_no"], [])
            # 증거 위계는 로드 시 계산해 붙인다. 재청킹하지 않는다 — 정규식
            # 몇 개라 로드 비용이 무시할 만하고, 규칙을 고칠 때마다 청크를 다시
            # 만들 이유가 없다. 청킹에 굽고 싶으면 chunk_docs.py에서 같은
            # 모듈(evidence_tier.annotate)을 부르면 된다.
            annotate(r)
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
        tier_on = use_evidence_tier() and bool(NUMERIC_QUESTION_RE.search(question))
        tier_mult = {}   # chunk_id → 적용된 위계 배율 (기각 서술의 재료)
        narrative_ok = bool(NARRATIVE_OK_QUESTION_RE.search(question or ""))

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
            # 추정 등급(tier_confident=False)으로는 순위를 흔들지 않는다.
            if tier_on and rec.get("tier_confident"):
                w = TIER_WEIGHT.get(rec.get("evidence_tier"), 1.0)
                if rec.get("evidence_tier") == 5 and narrative_ok:
                    w = max(w, TIER5_RELAXED)
                tier_mult[rec["chunk_id"]] = w
                mult *= w
            return mult

        return prior, active, tier_mult

    RESERVE = 3  # 사전확률 발동 시 해당 섹션 청크에 보장하는 최소 슬롯 수

    def search(self, question: str, topk=10, companies=None):
        comps = companies or self.route(question)
        if not comps:
            return {"companies": [], "hits": [], "priors": [],
                    "tier_demoted": [], "section_route": []}
        prior, active, tier_mult = self.make_prior(question)

        # 슬롯 보장은 '어느 절에 답이 있는가'를 지목하는 강한 프리어(가중치 0.6↑)에만 적용.
        # 약한 수식 프리어(연결/별도/보고서종류, 0.4)는 점수 배율에만 관여한다.
        strong = [sr for sr, w in active if w >= 0.6] or [sr for sr, _ in active]

        def is_prior_hit(rec):
            target = rec["section_path"] + " " + rec["report_nm"]
            return any(sec_re.search(target) for sec_re in strong)

        # 지정형 경로 — 지표어가 절에 매핑되면 그 절 안에서만 순위를 매긴다.
        # 매핑이 없거나 후보가 0건이면 즉시 탐색형(전체 검색)으로 되돌아간다.
        route_pat, route_word = ((None, None) if not use_section_route()
                                 else resolve_sections(question))

        per = max(topk // len(comps), 3)
        hits, demoted, route_notes = [], [], []
        for c in comps:
            idx = self.index_for(c)
            cands = []
            if route_pat:
                pool = [r for r in idx.records if section_matches(r, route_pat)]
                if pool:
                    allowed = {r["chunk_id"] for r in pool}

                    def scoped(rec, _p=prior, _a=allowed):
                        return _p(rec) if rec["chunk_id"] in _a else 0.0

                    cands = idx.search(question, topk=max(per * 6, 30), prior=scoped)
                    if cands:
                        route_notes.append(
                            f"[2+] 절 지정: '{route_word}' → {route_pat[2]} "
                            f"(후보 {len(pool)}개, {c})")
                    else:
                        route_notes.append(
                            f"[2!] 지정 절 후보에 질의어가 없음 → 전체 검색으로 전환 ({c})")
                else:
                    route_notes.append(f"[2!] 지정 절에 후보 없음 → 전체 검색으로 전환 ({c})")
            if not cands:
                cands = idx.search(question, topk=max(per * 6, 30), prior=prior)
            main = cands[:per]
            # 위계 가중치가 없었으면 상위에 들었을 하위 tier 청크. 밀려난 사실
            # 자체가 판단이므로 기각 서술의 재료로 남긴다.
            if tier_mult:
                raw = sorted(cands, key=lambda h: -(h[1] / tier_mult.get(h[0]["chunk_id"], 1.0)))
                kept = {h[0]["chunk_id"] for h in main}
                demoted += [h[0] for h in raw[:per]
                            if h[0]["chunk_id"] not in kept
                            and (h[0].get("evidence_tier") or 0) >= 5]
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
                "priors": [sr.pattern for sr, _ in active],
                "tier_demoted": demoted, "section_route": route_notes}
