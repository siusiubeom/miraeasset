"""저장된 실제 응답에 KAM 후처리만 재적용한다. 생성 재실행·정답률 평가가 아니다."""
import argparse
import json
import re
from pathlib import Path

import answerer as A


def replay(row):
    result = dict(row)
    A.begin_request([])
    context = row.get("retrieved_context", "")
    prose, separator, log = row.get("think_trace", "").partition("\n\n---\n[시스템 로그]\n")
    if not separator:
        return result
    trace = log.splitlines()
    hits = []
    for header in re.findall(r"===== \[근거 \d+\] (.*?) =====", context):
        parts = header.split(" | ")
        if len(parts) < 5:
            continue
        no = re.search(r"접수번호 (\d{14})", header)
        dt = re.search(r"접수일 (\d{4})-(\d{2})-(\d{2})", header)
        if not no or not dt:
            continue
        supersedes = re.search(r"정정본\(대상→([\d,]+)\)", header)
        superseded = re.search(r"정정으로 대체됨→([\d,]+)", header)
        hits.append((dict(corp=parts[0], report_nm=parts[1], rcept_no=no[1],
                          rcept_dt="".join(dt.groups()), section_path=parts[4].split(" ※")[0],
                          subtype="", text="", group="",
                          supersedes=supersedes[1].split(",") if supersedes else [],
                          superseded_by=superseded[1].split(",") if superseded else []), 0))
    A._REQ["evidence_hits"] = hits
    A._REQ["audit_context"] = context
    A._REQ["judgment_lines"] = ["- " + s.removeprefix("[판정근거] ")
                                for s in trace if s.startswith("[판정근거] ")]
    result["think_trace"] = A.merge_trace(prose, trace)
    result["kam_replayed"] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("output")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        p.error("기존 결과를 보존하기 위해 새 output 경로를 지정하세요")
    rows = json.loads(Path(args.source).read_text(encoding="utf-8"))
    out.write_text(json.dumps([replay(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"KAM 후처리 재생 {len(rows)}건 (LLM 호출 0회)")


if __name__ == "__main__":
    main()
