# -*- coding: utf-8 -*-
"""13문항 적대적 검증셋 재실행. 사용: py run_adversarial.py [출력파일]

질문 목록은 직전 실행 결과(adversarial_results.json)에서 그대로 가져와
개정 전후를 같은 문항으로 비교한다.
"""
import io, json, sys, time
from pathlib import Path
import answerer as A

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SRC = HERE / "adversarial_results_v13.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "adversarial_results_new.json"

qs = [(r["qid"], r["q"]) for r in json.loads(SRC.read_text(encoding="utf-8"))]
out = []
for qid, q in qs:
    t0 = time.time()
    A.begin_request()
    d = A.answer_question(qid, q)
    st = A.call_stats()
    out.append({"qid": qid, "q": q, "think_trace": d["think_trace"], "answer": d["answer"],
                "retrieved_context": d["retrieved_context"],
                "_sec": round(time.time() - t0, 2),
                "_calls": st["n_calls"], "_tpm": st["tpm_cost"]})
    print(f"[{qid}] {out[-1]['_sec']}s calls={out[-1]['_calls']}")
    print("   ", d["answer"].replace("\n", " ")[:200])
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved:", OUT)
