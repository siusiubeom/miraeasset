# -*- coding: utf-8 -*-
"""평가측 호출 방식 그대로(순차 GET) 서버를 테스트한다. 사용: py test_client.py [base_url]"""
import json, ssl, sys, time, urllib.parse, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"

# 이 PC의 Python CA 저장소에 만료된 루트가 있어 https 검증이 실패한다
# (curl은 정상). 검증을 기본으로 두되, 만료 오류일 때만 한 번 우회하고 경고한다.
# 우리 엔드포인트를 우리가 호출하는 스모크 테스트이므로 이 우회는 안전하다.
_CTX_INSECURE = ssl._create_unverified_context()


def fetch(url, timeout=300):
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
        print(f"  [경고] TLS 인증서 검증 실패({e.reason.verify_message}) — "
              f"검증 없이 재시도합니다. 이 PC의 CA 저장소 문제입니다.")
        return urllib.request.urlopen(url, timeout=timeout, context=_CTX_INSECURE)

TESTS = [
    ("Q-T01", "삼성전자가 2025년 7월 체결한 반도체 위탁생산 공급계약의 계약상대는 어디인가?"),
    ("Q-T02", "현대차의 2025년 연결기준 매출액은 얼마인가?"),
    ("Q-T03", "알테오젠의 최대주주는 누구인가?"),
    ("Q-T04", "삼성전자의 2026년 3분기 실적 전망은 어떻게 되는가?"),      # 정보한계
    ("Q-T05", "카카오 주가가 앞으로 오를까요?"),                          # 투자의견 → 한계
    ("Q-T06", "테슬라의 2025년 매출액은 얼마인가?"),                      # 코퍼스 외 회사
    ("Q-T01", "삼성전자가 2025년 7월 체결한 반도체 위탁생산 공급계약의 계약상대는 어디인가?"),  # 캐시(멱등) 확인
]

for qid, q in TESTS:
    url = f"{BASE}/answer?" + urllib.parse.urlencode({"question_id": qid, "question": q})
    t0 = time.time()
    with fetch(url) as r:
        d = json.loads(r.read().decode("utf-8"))
        code = r.status
    dt = time.time() - t0
    assert set(d) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
    assert all(isinstance(v, str) for v in d.values())
    print(f"--- {qid}  HTTP {code}  {dt:.2f}s  ctx={len(d['retrieved_context'])}자")
    print("  trace:", d["think_trace"].replace("\n", " / ")[:220])
    print("  answer:", d["answer"].replace("\n", " ")[:220])
