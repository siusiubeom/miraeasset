"""비교표의 기간·단위·중복과 사건 연결의 반례 검사."""
import unittest
from unittest.mock import patch
from decimal import Decimal
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from comparative_reports import observations,pick,run
from event_records import contract_graph,purpose_values
from kam_report import render


class ComparisonTests(unittest.TestCase):
    def test_fiscal_columns_and_negative(self):
        text='''| 제 9 기 2025.01.01 부터 2025.12.31 까지 |
| 제 8 기 2024.01.01 부터 2024.12.31 까지 |
| (단위 : 백만원) |
| | 제 9 기 | 제 8 기 |
| 영업이익 (주3) | (25) | 100 |'''
        obs=observations(text,2025)
        self.assertEqual(pick(obs,'영업이익',2025).value,Decimal(-25))
        self.assertEqual(pick(obs,'영업이익',2024).value,Decimal(100))

    def test_unknown_fiscal_not_assumed_current(self):
        self.assertEqual(observations('| (단위 : 원) |\n| | 제 8 기 |\n| 매출액 | 300 |',2025),[])

    def test_explicit_annual_header_ignores_percentage_columns(self):
        obs=observations('| (단위 : 백만원) |\n| 구분 | 제9기 2025년 연간 | 비율 | 8기 2024년연간 | 비율 |\n| 연결 후 매출액 합계 | 120 | 100% | 100 | 100% |',2025)
        self.assertEqual(pick(obs,'연결후매출액합계',2024).value,100)
        self.assertEqual(len(obs),2)

    def test_block_keeps_basic_and_diluted_separate(self):
        text='''| 기본 EPS | 기본 EPS |
| 당기 | (단위 : 원) |
| 기본주당이익 | 10 |
| 전기 | (단위 : 원) |
| 기본주당이익 | 8 |
| 희석 EPS | 희석 EPS |
| 당기 | (단위 : 원) |
| 희석주당이익 | 9 |'''
        obs=observations(text,2024)
        self.assertEqual(pick(obs,'기본주당이익',2023).value,8)
        self.assertEqual(pick(obs,'희석주당이익',2024).value,9)

    def test_conflicting_values_are_not_selected(self):
        obs=observations('| 당기 | (단위 : 원) |\n| 매출액 | 10 |\n| 매출액 | 11 |',2024)
        self.assertIsNone(pick(obs,'매출액',2024))

    def test_new_unlabelled_block_cannot_inherit_period(self):
        obs=observations('| 당기 | (단위 : 원) |\n| 매출액 | 10 |\n| 다른 표 | 다른 표 |\n| 매출액 | 99 |',2024)
        self.assertEqual(pick(obs,'매출액',2024).value,10)

    def test_inline_eps_unit_does_not_change_statement_unit(self):
        obs=observations('| 당기 | (단위 : 백만원) |\n| 기본주당이익 (단위 : 원) | 10 |\n| 매출액 | 99 |',2024)
        self.assertEqual(pick(obs,'매출액',2024).unit,'백만원')

    def test_failed_supported_comparison_does_not_fall_through(self):
        with patch('comparative_reports.annual',return_value=None):
            result=run('기업 2024년과 2025년 기본주당이익과 가중평균 주식수를 비교','기업')
        self.assertIsNotNone(result)
        self.assertIn('확정하지 못했습니다',result.answer)
        self.assertEqual(result.sources,[])

    def test_unrelated_question_does_not_claim_comparison_failure(self):
        self.assertIsNone(run('기업 배당 기준일','기업'))


def source(no,dt,name='계약A',party='회사X',target=None):
    text=f'| - 체결계약명 | {name} |\n| 3. 계약상대 | {party} |'
    if target:
        text+=f'\n| 2. 정정관련 공시서류제출일 | {target} |'
    return dict(rcept_no=no,rcept_dt=dt,report_nm=('정정 ' if target else '')+'공급계약',text=text)


class EventTests(unittest.TestCase):
    def test_purpose_labels_and_values_come_from_same_clause(self):
        values=purpose_values('2조 300억원은 임직원 보상을 목적으로 하며, 1조 500억원은 주주가치 제고의 목적으로 한다.')
        self.assertEqual(values,[('임직원 보상',Decimal(2030000000000)),('주주가치 제고',Decimal(1050000000000))])

    def test_same_day_distinct_contracts_not_joined(self):
        src=[source('1','20230101'),source('2','20230101',name='계약B'),source('3','20230201',target='2023-01-01')]
        _,_,parents,_,unresolved=contract_graph(src)
        self.assertEqual(parents,{'3':'1'})
        self.assertEqual(unresolved,[])

    def test_ambiguous_original_not_chosen_by_receipt_order(self):
        src=[source('1','20230101'),source('2','20230101'),source('3','20230201',target='2023-01-01')]
        _,_,parents,_,unresolved=contract_graph(src)
        self.assertEqual(parents,{})
        self.assertEqual(unresolved,['3'])

    def test_missing_original_remains_partial_but_children_connect(self):
        src=[source('2','20230201',target='2023-01-01'),source('3','20230301',target='2023-02-01')]
        _,_,parents,groups,unresolved=contract_graph(src)
        self.assertEqual(parents,{'3':'2'})
        self.assertEqual(groups['2'],['2','3'])
        self.assertEqual(unresolved,['2'])

    def test_kam_uses_actual_basis_and_does_not_create_rejections(self):
        report=render({'audit_basis':'동일 보고서 비교표시를 사용했다','audit_result':'합계 10원과 일치했다','audit_limit':'다른 판본은 대사하지 않았다'},[])
        self.assertIn('동일 보고서 비교표시',report)
        self.assertIn('합계 10원',report)
        self.assertNotIn('기각',report)


if __name__=='__main__':
    unittest.main()
