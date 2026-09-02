# -*- coding: utf-8 -*-
"""같은 질문 N회 반복 실행 → 답변 동일 여부. 사용: py repeat_check.py QID [N]"""
import json, sys
from pathlib import Path
import answerer as A
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
qs = {r["qid"]: r["q"] for r in json.loads(
    (HERE / "adversarial_results_v13.json").read_text(encoding="utf-8"))}
qid = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
answers = []
for i in range(n):
    A.begin_request()
    answers.append(A.answer_question(qid, qs[qid])["answer"])
    print(f"--- {i+1}회차: {answers[-1][:160]}")
print(f"\n[{qid}] 동일 여부: {len(set(answers)) == 1} (서로 다른 답변 {len(set(answers))}종)")
