import unittest
from unittest.mock import patch
from dataclasses import replace
from decimal import Decimal

from accounting_metrics import Fact, operating_margin, derive_margins
from aggregate_tools import correction_changes
import aggregate_tools as AGG
from table_cells import parse_cells


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.profit = Fact("삼성전자", "연결", "2025", "annual", "영업이익", "DS",
                           "백만원", Decimal("24858075"), "20260310002820", "table1")
        self.revenue = replace(self.profit, metric="매출액", value=Decimal("130128162"))

    def test_operating_margin_not_profit_share(self):
        self.assertEqual(operating_margin(self.profit, self.revenue), Decimal("19.10"))
        whole_p = replace(self.profit, segment="전사", value=Decimal("43601051"))
        whole_r = replace(self.revenue, segment="전사", value=Decimal("333605938"))
        self.assertEqual(operating_margin(whole_p, whole_r), Decimal("13.07"))

    def test_incompatible_coordinates_block_calculation(self):
        for key, value in dict(entity="다른회사", basis="별도", period="2024", duration="quarter",
                               segment="전사", unit="억원", source="다른판본", table="다른표").items():
            with self.subTest(key=key):
                self.assertIsNone(operating_margin(self.profit, replace(self.revenue, **{key: value})))

    def test_unknown_zero_and_wrong_metric(self):
        for revenue in [replace(self.revenue, value=Decimal(0)),
                        replace(self.revenue, metric="자본총계")]:
            self.assertIsNone(operating_margin(self.profit, revenue))
        self.assertIsNone(operating_margin(replace(self.profit, basis="unknown"),
                                          replace(self.revenue, basis="unknown")))

    def test_header_continuation_retains_logical_table(self):
        text = """| 당기 | (단위 : 백만원) |
| 구분 | DS 부문 | DX 부문 | 기업 전체 합계 |
| 매출액 | 130128162 | 187967346 | 333605938 |

| 구분 | 기업 전체 | 기업 전체 | 기업 전체 |
| 영업이익 | 24858075 | 12852650 | 43601051 |
"""
        section = "III > 30. 부문별 보고 (연결)"
        cells = [c._replace(table_id=c.table_id + ":" + section)
                 for c in parse_cells(text, "20260310002820", "20260310", "사업보고서 (2025.12)")]
        hits = [(dict(rcept_no="20260310002820", corp="삼성전자", section_path=section,
                      report_nm="사업보고서 (2025.12)"), 1)]
        results = derive_margins(cells, "2025년 DS 매출 비중과 영업이익 비중", hits)
        self.assertEqual({d.value for d in results}, {Decimal("19.10"), Decimal("13.07")})
        relation = next(d for d in results if d.kind == "margin_relation")
        self.assertIn("19.10%는 전사 13.07%보다 높다", relation.desc)

    def test_changed_fields_count_not_document_count(self):
        text = """| 정정항목 | 정정전 | 정정후 |
|---|---|---|
| 5. 계약기간 - 종료일 | 2024-12-31 | 2025-03-31 |
| 2. 계약내역- 계약금액(원) | 821,260,000,000 | 1,189,540,000,000 |
| 2. 계약내역- 매출액대비(%) | 10.97 | 15.89 |
| 계약상대 | A | A |
"""
        changes = correction_changes(text + "\n" + text)
        self.assertEqual(len(changes), 3)
        self.assertEqual({c["category"] for c in changes if c["category"]}, {"계약금액", "계약기간"})
        with patch.object(AGG, "docs_for", return_value=[dict(rcept_no="doc1", rcept_dt="20241230")]), \
             patch.object(AGG, "_doc_text", return_value=text + "\n" + text):
            result = AGG.distinct_count("회사")
        self.assertEqual(result["n_docs"], 1)
        self.assertEqual(result["change_occurrences"], {"계약금액": 1, "계약기간": 1})

    def test_reason_alone_does_not_prove_changed_fields(self):
        self.assertEqual(correction_changes("| 정정사유 | 계약금액 및 계약기간 변경 |"), [])

    def test_changes_without_reason_keep_source_and_coverage(self):
        docs = [dict(rcept_no="doc1", rcept_dt="20241230")]
        text = "| 정정항목 | 정정전 | 정정후 |\n| 계약금액 | 100 | 200 |"
        with patch.object(AGG, "docs_for", return_value=docs), \
             patch.object(AGG, "_doc_text", return_value=text):
            result = AGG.distinct_count("회사")
        self.assertEqual(result["n_parsed"], 0)
        self.assertEqual(result["change_occurrences"], {"계약금액": 1})
        self.assertEqual(result["per_doc"][0]["rcept_no"], "doc1")
        self.assertEqual(result["n_unresolved_change_docs"], 0)


if __name__ == "__main__":
    unittest.main()
