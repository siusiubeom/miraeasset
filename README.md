# 공시 Agent — 제10회 2026 미래에셋증권 AI Festival

DART 공시 코퍼스(70개사, 4,204건) 기반 질의응답 Agent.
질문을 받아 관련 공시를 검색하고, HyperCLOVA X(HCX-005)로 근거 기반 답변을 생성합니다.

---

## 평가용 API End-point (필수 제출 항목)

> **End-point:** `https://preoccupy-distort-jujitsu.ngrok-free.dev/answer`
>
> HTTPS 443 표준 포트이므로 포트 표기를 생략합니다. 개인 환경에서 서버를 운영하고
> ngrok 고정 도메인으로 외부에 노출했습니다 — 평가 규격이 명시적으로 허용하는 방식입니다
> (평가API_규격_및_QA.md: "개인 환경 구성 시 ngrok 등 터널링 서비스 자유롭게 활용 가능").

### 호출 방법 (주최측 평가 규격)

```
GET {end-point}/answer?question_id={질의 ID}&question={평가 질의}
```

```bash
curl -G "https://preoccupy-distort-jujitsu.ngrok-free.dev/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=삼성전자의 2025년 연결기준 매출액은 얼마인가?"
```

### 응답 형식 (`application/json`, 모든 필드 string)

```json
{
  "question_id": "Q-001",
  "question": "평가 질의 원문",
  "retrieved_context": "답변 생성에 참고한 검색 문서 (여러 문서는 ===== [근거 N] ===== 구분자로 연결)",
  "think_trace": "사고·추론·도구 사용 과정",
  "answer": "최종 생성 답변"
}
```

- 어떤 내부 오류에도 **5xx 없이 항상 200 + 규격 JSON**으로 응답합니다.
- 문항당 300초 타임아웃 대비, 내부 285초 가드 후 부분 응답을 반환합니다.
- 동일 `question_id` 재수신(주최측 재시도) 시 캐시된 동일 응답을 반환합니다(멱등).
- 상태 확인용으로 `GET /health` 를 추가 제공합니다.

---

## 실행 방법

### 0. 환경

- **Python 3.13** (3.10+ 호환), **외부 패키지 의존성 없음** — 표준 라이브러리만 사용 (`requirements.txt` 참고)
- API 키 설정: `.env.example` 을 `.env` 로 복사 후 `CLOVA_API_KEY` 입력
  (미설정 시 생성 단계는 추출식 폴백으로 동작)

### 1. 데이터 배치

주최측 제공 코퍼스를 `corpus/` 에 배치합니다 (git에는 포함되어 있지 않음 — 아래 [데이터·산출물] 참고).

```
corpus/
├── manifest.jsonl     # 문서 목록 4,204행
├── universe.csv       # 기업 마스터 70행
└── raw/               # 공시 원문 XML (periodic / major / exchange / holding)
```

### 2. 전처리 (원본 XML → Markdown → 청크)

```bash
python preprocess/build_company_md.py     # corpus/raw XML → processed/companies, processed/docs (기업별 md)
python baseline/chunk_docs.py             # processed/docs → processed/chunks/*.jsonl (검색용 청크)
```

이미 생성된 산출물(`processed/`, 3.6GB)은 아래 클라우드 스토리지 링크로도 제공합니다.

### 3. API 서버 기동

```bash
python baseline/server.py 80        # 운영: 80 포트 (평가 규격)
python baseline/server.py           # 로컬 개발: 기본 8080 포트
```

### 4. 동작 확인

```bash
python baseline/test_client.py http://127.0.0.1:8080
# 평가측과 동일한 순차 GET 호출로 7문항(정상/정보한계/투자의견/코퍼스 외/캐시) 검증
```

---

## 프로젝트 구조

```
mirae/
├── README.md                  # 본 문서 (평가용 End-point 명시)
├── requirements.txt           # 의존성 없음 명시 (재현성)
├── .env.example               # API 키 템플릿 (.env 로 복사해 사용)
├── preprocess/
│   ├── dart_to_md.py          # DART XML → Markdown 변환기
│   └── build_company_md.py    # 기업별 문서 빌드 (병렬)
├── baseline/
│   ├── retrieval.py           # 회사 라우팅 + BM25 검색 + 정정공시 우선 처리
│   ├── answerer.py            # 검색 → HCX-005 생성 파이프라인 (PII 마스킹, 투자의견 차단)
│   ├── server.py              # 평가용 /answer API 서버 (stdlib http.server)
│   ├── test_client.py         # 평가 방식 그대로 검증하는 클라이언트
│   ├── chunk_docs.py          # 문서 청킹
│   └── run_baseline.py        # 검색 품질 자체 평가 (12문항)
├── corpus/                    # (git 제외) 주최측 제공 원본 — 5.3GB
└── processed/                 # (git 제외) 전처리 산출물 — 3.6GB, 재생성 가능
```

## 데이터·산출물 (대용량 — git 미포함)

| 항목 | 크기 | 제공 방법 |
|---|---|---|
| `corpus/` 원본 공시 | 5.3GB | 주최측 배포본 사용 (재배포하지 않음) |
| `processed/` 전처리 산출물 | 3.6GB | **TODO: 클라우드 스토리지 링크 기입** — 또는 위 [실행 방법 2]로 재생성 (약 N분 소요) |

## 팀 문서

- [공시Agent_대회요약.md](공시Agent_대회요약.md) — 과제·데이터 분석 요약
- [평가API_규격_및_QA.md](평가API_규격_및_QA.md) — 평가 규격 및 운영진 Q&A 정리
