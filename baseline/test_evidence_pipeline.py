"""근거 출처·정정 상태·구조 경계 회귀. 외부 API 호출 없음."""
import gzip
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import answerer as A
import aggregate_tools as AGG
import struct_ops as SO
from form_blocks import extract_blocks
from table_cells import parse_cells


def record(no="20241118000328", section="자기주식", text=""):
    return dict(rcept_no=no, corp="삼성전자", rcept_dt=no[:8],
                report_nm="주요사항보고서(자기주식취득결정)",
                section_path=section, subtype="", group="major", text=text,
                chunk_id=no + section, supersedes=[], superseded_by=[])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        A.begin_request([])

    def test_unloaded_is_not_absent(self):
        text = A.build_judgment_log([(record(), 1)], [])
        self.assertIn("미로딩", text)
        self.assertNotIn("정정본 없음", text)

    def test_intermediate_correction_is_not_terminal(self):
        r = record()
        r.update(supersedes=["20241115000375"], superseded_by=["20241119000328"])
        text = A.build_judgment_log([(r, 1)], [])
        self.assertIn("중간 정정본", text)
        self.assertNotIn("말단이므로", text)
        self.assertNotIn("근거에서 제외했다", text)

    def test_invalid_trace_is_blocked_but_known_source_is_preserved(self):
        log = []
        A.build_judgment_log([(record(), 1)], log)
        good = "취득결정 확인\n접수번호 20241118000328의 취득예정금액을 확인했다."
        self.assertIn("20241118000328", A.merge_trace(good, log).split("[시스템 로그]")[0])
        bad = A.merge_trace("취득결정 확인\n접수번호 20990101000001을 채택했다.", log)
        self.assertNotIn("20990101000001", bad)
        self.assertIn("KAM-차단", bad)

    def test_formula_keeps_all_terms_and_source(self):
        text = "산출근거\n① 보통주식 100주\n② 기타주식 200주\n③ 비교값 300주\n\n다른 설명"
        blocks = extract_blocks([(record(text=text), 1)])
        self.assertIn("20241118000328", blocks)
        self.assertIn("③ 비교값 300주", blocks)
        self.assertNotIn("다른 설명", blocks)

    def test_kam_without_source_uses_observations(self):
        log = []
        A.build_judgment_log([(record(), 1)], log)
        text = A.merge_trace("일반적인 공시 검토\n종합적으로 검토했다.", log)
        self.assertIn("20241118000328", text.split("[시스템 로그]")[0])
        self.assertIn("KAM-전환", text)
        self.assertNotIn("종합적으로 검토했다", text)

    def test_gzip_matches_plain_and_missing_is_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r = record(text="공시 원문")
            with gzip.open(root / "삼성전자.jsonl.gz", "wt", encoding="utf-8") as f:
                f.write(json.dumps(r))
            with patch.object(AGG, "CHUNK_DIR", root), patch.object(AGG, "_chunk_cache", AGG.OrderedDict()):
                self.assertEqual(AGG.chunks_by_rcept("삼성전자")[r["rcept_no"]][0], r)
                with self.assertRaises(FileNotFoundError):
                    AGG.chunks_by_rcept("없는회사")

    def test_section_ids_prevent_cross_table_sum(self):
        a = record(section="A", text="| 2. 취득예정금액(원) | 보통주식 | 100 |")
        b = record(section="B", text="| 2. 취득예정금액(원) | 기타주식 | 200 |")
        with patch.object(AGG, "chunks_by_rcept", return_value={a["rcept_no"]: [a, b]}):
            cells = A.struct_cells_for_hits([(a, 1), (b, 1)])
        self.assertEqual(len({c.table_id for c in cells}), 2)
        self.assertFalse(SO.derive(cells, "취득예정금액"))

    def test_neighbor_evidence_is_returned(self):
        a = record(text="| 2. 취득예정금액(원) | 보통주식 | 2682737598000 |")
        b = dict(a, chunk_id="next", text="| 2. 취득예정금액(원) | 기타주식 | 317262452400 |")
        with patch.object(AGG, "chunks_by_rcept", return_value={a["rcept_no"]: [a, b]}):
            cells = A.struct_cells_for_hits([(a, 1)])
        results = SO.derive(cells, "취득예정금액")
        self.assertEqual(results[0].value, Decimal("3000000050400"))
        self.assertIn("317262452400", A.build_context(A.response_evidence([(a, 1)])))

    def test_plain_unit_declaration(self):
        cells = parse_cells("(단위 : 백만원)\n| 구분 | DS | 전사 |\n| 매출액 | 40 | 100 |",
                            "doc", "20260301", "사업보고서 (2025.12)")
        self.assertEqual({c.unit for c in cells}, {"백만원"})

    def test_previous_previous_period(self):
        cells = parse_cells("| 전전기 | (단위 : 원) |\n| 구분 | DS | 전사 |\n| 매출액 | 40 | 100 |",
                            "doc", "20260301", "사업보고서 (2025.12)")
        self.assertEqual({c.period for c in cells}, {"2023"})

    def test_aggregate_missing_counts_are_restored(self):
        res = dict(n_docs=8, n_parsed=8, missing=0, counts={"계약금액 변경": 8},
                   top=["계약금액 변경"], top_n=8, tied=False,
                   per_doc=[dict(rcept_no="20241118000328", reason="계약금액 변경")])
        with patch.object(A, "run_aggregate", return_value=res), \
             patch.object(A, "aggregate_context", return_value="계약금액 변경"), \
             patch.object(A, "clova_available", return_value=True), \
             patch.object(A, "call_clova_raw", return_value="[판단]\n소제목: 사유 집계\n사유를 집계했다.\n[답변]\n최다 사유는 계약금액 변경입니다."):
            result = A.answer_aggregate("L7", "삼성E&A 2024년 공급계약 정정 사유별 집계", "삼성E&A", None, [])
        self.assertIn("계약금액 변경: 8건", result["answer"])
        self.assertIn("사유 미확인 0건", result["answer"])


if __name__ == "__main__":
    unittest.main()
