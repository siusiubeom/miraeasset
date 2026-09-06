"""서식 연산, 출처 검증, 요청 상태 분리를 검사한다."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal
import threading
import unittest
from unittest.mock import patch

import answerer as A
import disclosure_tools as D
from kam_report import render


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        A.begin_request([])

    def test_request_threads_do_not_share_state(self):
        barrier = threading.Barrier(2)
        def work(n):
            A.begin_request([])
            A._REQ['calls'].append(n)
            A._REQ['audit_result'] = str(n)
            barrier.wait(timeout=5)
            return A._REQ['calls'], A._REQ['audit_result']
        with ThreadPoolExecutor(2) as pool:
            got = list(pool.map(work, [1,2]))
        self.assertEqual(got, [([1],'1'),([2],'2')])
        self.assertEqual(A._REQ['calls'], [])

    def test_new_request_discards_previous_evidence(self):
        A._REQ['audit_docs'] = [{'rcept_no':'previous'}]
        A.begin_request([])
        self.assertNotIn('audit_docs', A._REQ)

    def test_kam_uses_observed_result_for_aggregate(self):
        state = dict(audit_subject='계약 정정',audit_docs=[dict(rcept_no='20241230800946',report_nm='정정공시')],
                     audit_result='문서 1건에서 두 변경 범주를 확인했다',audit_limit='해당 문서의 변경표에 한정한다')
        text = render(state, [])
        self.assertIn('20241230800946',text)
        self.assertIn('문서 1건',text)
        self.assertNotIn('기각',text)
        self.assertLessEqual(len(text.splitlines()),5)

    def test_cb_field_conflict_is_not_first_value(self):
        value,_=D.unique_field('| 전환가액 (원/주) | 10 | 10 |\n| 전환가액 (원/주) | 20 |',['전환가액(원/주)'])
        self.assertIsNone(value)

    def test_date_parser_validates_calendar(self):
        self.assertEqual(D.date_values('2024년 2월 29일과 2023-02-29'),[date(2024,2,29)])

    def test_amount_after_month_is_not_a_day(self):
        self.assertFalse(A._FULL_DATE_RE.search('2025년 2월 2,696,357,832,000원'))
        self.assertEqual(A._ymd(A._FULL_DATE_RE.search('2025년 2월 18일')), '20250218')
        self.assertEqual(A._ymd(A._FULL_DATE_RE.search('2025-02-18')), '20250218')

    def test_blocked_prose_does_not_contradict_valid_derivations(self):
        result=A.attach_derivations(A.VERIFY_BLOCK_MSG,['100 - 20 = 80원'],[],[],docs=['20230101000001'])
        self.assertNotIn('수치를 확정하지 못했습니다',result)
        self.assertIn('100 - 20 = 80원',result)
        self.assertTrue(A._REQ['audit_exclusions'])

    def test_financial_sign(self):
        self.assertEqual(D.number('(2,795)'),Decimal('-2795'))
        self.assertEqual(D.number('-'),Decimal(0))
        self.assertIsNone(D.number('약 2,795'))

    def cb_source(self):
        return dict(corp='테스트',report_nm='전환사채권발행결정',rcept_no='20230101000001',rcept_dt='20230101',text='''
| 회차 | 7 | 종류 | CB |
| 2. 사채의 권면(전자등록)총액 (원) | 1,001 |
| 전환가액 (원/주) | 10 |
| 최저 조정가액 (원) | 6 |
| 전환청구기간 | 시작일 | 2024-01-01 |
| 전환청구기간 | 종료일 | 2028-01-01 |
| 사채만기일 | 2028-02-01 |
1주 미만 단주는 현금으로 지급한다.
''')

    def call_cb(self,question,source=None):
        with patch.object(D.AGG,'docs_for',return_value=[{}]), patch.object(D,'document',return_value=source or self.cb_source()):
            return D.cb_terms(question,'테스트')

    def test_cb_floor_and_cash_remainder(self):
        result=self.call_cb('테스트 제7회 CB 전환가액 하한 전환가능주식수')
        self.assertIn('166주',result.answer)
        self.assertIn('66주 증가',result.answer)
        self.assertIn('5원',result.answer)
        self.assertIn('실제 리픽싱',result.answer)

    def test_cb_interval_end_inclusive_and_not_maturity(self):
        result=self.call_cb('테스트 제7회 CB 2028-01-01 전환청구기간 날짜 조건')
        self.assertIn('포함됩니다',result.answer)
        result=self.call_cb('테스트 제7회 CB 2028-01-02 전환청구기간 날짜 조건')
        self.assertIn('포함되지 않습니다',result.answer)

    def test_cb_other_series_does_not_substitute(self):
        self.assertIsNone(self.call_cb('테스트 제8회 CB 전환가액 하한 전환가능주식수'))

    def test_cb_unknown_fractional_rule_blocks_shares(self):
        source=self.cb_source()
        source['text']=source['text'].replace('1주 미만 단주는 현금으로 지급한다.','')
        self.assertIsNone(self.call_cb('테스트 제7회 CB 전환가액 하한 전환가능주식수',source))

    def rollforward(self, closing='125', path='관계기업 (연결)'):
        meta=dict(rcept_no='20260301000001',rcept_dt='20260301',report_nm='사업보고서 (2025.12)',doc_group='periodic')
        text=f'''1) 제 10(당) 기
| (단위: 백만원) |
| 회사명 | 기초 | 취득 | 배당금수령 | 지분법손익 | 기타 | 기말 |
| 관계회사 | 100 | 10 | (5) | 20 | - | {closing} |
'''
        rec=dict(meta,corp='테스트',group='periodic',section_path=path,text=text)
        with patch.object(D.AGG,'docs_for',return_value=[meta]), \
             patch.object(D.AGG,'chunks_by_rcept',return_value={meta['rcept_no']:[rec]}):
            return D.equity_rollforward('테스트 2025년 연결재무제표 관계회사 지분법 배당이 별도 수익인지','테스트')

    def test_equity_rollforward_preserves_dividend_sign(self):
        result=self.rollforward()
        self.assertIn('100 + 10 − 5 + 20 + 0 = 125백만원',result.answer)
        self.assertIn('연결 관계기업',result.answer)

    def test_unreconciled_or_wrong_basis_rollforward_is_rejected(self):
        self.assertIsNone(self.rollforward(closing='126'))
        self.assertIsNone(self.rollforward(path='관계기업 (별도)'))


if __name__ == '__main__':
    unittest.main()
