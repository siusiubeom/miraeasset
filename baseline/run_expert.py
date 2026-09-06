"""질문 파일을 순차 반복 실행하고 매 응답을 저장한다. 정답률과 표현 변동은 별개다."""
import argparse
import json
import time
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import answerer as A


def load_questions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("results", data.get("rows", [])) if isinstance(data, dict) else data
    return [{"qid": r.get("qid") or r.get("question_id"),
             "question": r.get("question") or r.get("q")} for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--ids", help="쉼표로 구분한 문항 ID")
    args = p.parse_args()
    if args.repeat < 1:
        p.error("--repeat must be positive")
    if not A.clova_available():
        p.error("CLOVA_API_KEY가 없어 실제 생성 품질을 측정할 수 없습니다")
    rows = load_questions(args.source)
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["qid"] in wanted]
        if wanted - {r["qid"] for r in rows}:
            p.error("요청한 문항 ID가 입력에 없습니다")
    if not rows or any(not r["qid"] or not r["question"] for r in rows):
        p.error("질문 파일이 비었거나 ID/질문이 누락되었습니다")
    output = Path(args.output)
    if output.exists():
        p.error("기존 측정 결과를 보존하기 위해 새 output 경로를 지정하세요")
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source, "repeat": args.repeat, "ids": args.ids,
        "source_sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        "code_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in sorted(Path(__file__).parent.glob("*.py"))},
        "flags": {k: os.environ.get(k, "default") for k in
                  ("USE_STRUCTURED_KAM", "USE_DISCLOSURE_TOOLS", "USE_ACCOUNTING_METRICS", "USE_CORRECTION_OCCURRENCES")},
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    for row in rows:
        for run in range(1, args.repeat + 1):
            t0 = time.monotonic()
            response = A.answer_question(row["qid"], row["question"])
            response.update(qid=row["qid"], run=run, sec=round(time.monotonic() - t0, 2),
                            calls=A.call_stats())
            results.append(response)
            output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{row['qid']} run={run} sec={response['sec']}", flush=True)
            if "[재시도-실패]" in response["think_trace"]:
                raise SystemExit("API 호출 실패: 결과를 저장하고 반복 실행을 중단했습니다")


if __name__ == "__main__":
    main()
