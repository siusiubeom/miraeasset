# -*- coding: utf-8 -*-
"""산술·어림수·출처 검증 회귀 테스트 (CLOVA 호출 없이 순수 헬퍼만 검사).

실행: python baseline/test_arith.py
"""
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from answerer import (format_krw, has_round_number, is_round_number, missing_rcept_no,
                      parse_krw, parse_krw_all, sum_krw, verify_number, verify_trace)

RESULTS = []


def check(case: str, expected, actual):
    ok = expected == actual
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {case}\n       기대={expected!r}\n       실제={actual!r}")


# ── 케이스 1: 합산 (자기주식 취득 3조원 — 부동소수점이면 뭉개지는 자릿수) ──────
EXT_SUM = """값: 2,682,737,598,000원 | 기준: 2025년 연결 | 출처: 주요사항보고서(자기주식취득결정), 2025-11-14, 20251114000123
값: 317,262,452,400원 | 기준: 2025년 연결 | 출처: 주요사항보고서(자기주식취득결정), 2025-11-14, 20251114000123"""

vals = parse_krw_all(EXT_SUM)
check("합산: 값 2건 파싱", [Decimal("2682737598000"), Decimal("317262452400")], vals)
total, _ = sum_krw(vals)
check("합산: Decimal 합계", Decimal("3000000050400"), total)
check("합산: 어림수 아님", False, has_round_number(EXT_SUM))
check("합산: 검산 통과(정답 인용)", True,
      verify_number("자기주식 취득 총액은 3,000,000,050,400원입니다.", total))
check("합산: 검산 실패(LLM 오답 인용)", False,
      verify_number("자기주식 취득 총액은 2,700,090,450,400원입니다.", total))
check("합산: 검산 trace 문구", "[검산] 불일치 감지: 코드값 3,000,000,050,400, 답변값 2,700,090,450,400",
      verify_trace("총액은 2,700,090,450,400원입니다.", total))

# ── 케이스 2: 단순 조회 (매출액 — 백만원 단위 표 기재값) ──────────────────────
EXT_ONE = ("값: 333,605,938백만원 | 기준: 2025년 연결기준 | "
           "출처: 삼성전자 사업보고서, 2026-03-10, 20260310000456")

check("단순조회: 원 단위 환산", Decimal("333605938000000"), parse_krw(EXT_ONE))
check("단순조회: 어림수 아님", False, has_round_number(EXT_ONE))
check("단순조회: 접수번호 존재", False, missing_rcept_no(EXT_ONE))
check("단순조회: 포맷", "333,605,938,000,000원 (약 333.61조원)", format_krw(parse_krw(EXT_ONE)))

# ── 케이스 3: 어림수 거부 (334조 원 → 코드 환산 금지) ─────────────────────────
EXT_ROUND = "값: 334조 원 | 기준: 2025년 연결기준 | 출처: 삼성전자 사업보고서 (2025.12)"

check("어림수: 유효숫자 3자리 판정", True, is_round_number("334"))
check("어림수: 끝자리 0 채움 판정", True, is_round_number("334,000,000,000,000"))
check("어림수: 정상값 오탐 없음", False, is_round_number("333,605,938"))
check("어림수: has_round_number", True, has_round_number(EXT_ROUND))
check("어림수: parse_krw는 None (환산 생략)", None, parse_krw(EXT_ROUND))
check("어림수: parse_krw_all 비어 있음", [], parse_krw_all(EXT_ROUND))
check("어림수: 출처 접수번호 누락 감지", True, missing_rcept_no(EXT_ROUND))

# ── 요약 ──────────────────────────────────────────────────────────────────────
failed = RESULTS.count(False)
print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} 통과" + (f", {failed} 실패" if failed else ""))
sys.exit(1 if failed else 0)
