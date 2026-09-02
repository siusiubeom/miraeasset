# -*- coding: utf-8 -*-
"""연속 호출 안정성 · TPM 소비 측정.

페이싱 없이 연속 호출해 429/5xx 재시도 동작과 질문당 토큰 소비를 집계한다.
실행: python baseline/measure_tpm.py [반복수] [라벨]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import answerer as a

QUESTIONS = [
    "삼성전자의 2024년 11월 자기주식 취득 결정 금액은?",
    "삼성전자의 2025년 연결기준 매출액은?",
]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    label = sys.argv[2] if len(sys.argv) > 2 else "run"
    rows = []
    t_start = time.time()
    for i in range(n):
        q = QUESTIONS[i % len(QUESTIONS)]
        t0 = time.time()
        r = a.answer_question(f"M-{i:02d}", q)
        stats = a.call_stats()
        t = r["think_trace"]
        rows.append({
            "i": i, "q": q[:20],
            "gen_err": any(l.startswith("[5-err]") for l in t.splitlines()),
            "parse_fail": "[5!] JSON 파싱 실패" in t,
            "retries": sum(1 for l in t.splitlines() if l.startswith("[재시도]")),
            "retry_failed": any(l.startswith("[재시도-실패]") for l in t.splitlines()),
            "budget_stop": any(l.startswith("[재시도-중단]") for l in t.splitlines()),
            "n_calls": stats["n_calls"], "tpm": stats["tpm_cost"],
            "sec": round(time.time() - t0, 1),
        })
        print(f"{i + 1:2}/{n} calls={rows[-1]['n_calls']} tpm={rows[-1]['tpm']:,} "
              f"retries={rows[-1]['retries']} err={rows[-1]['gen_err']} "
              f"{rows[-1]['sec']}s", flush=True)

    ok = sum(1 for r in rows if not r["gen_err"] and not r["parse_fail"])
    tot_retry = sum(r["retries"] for r in rows)
    avg_calls = sum(r["n_calls"] for r in rows) / len(rows)
    avg_tpm = sum(r["tpm"] for r in rows) / len(rows)
    summary = {
        "label": label, "n": n, "ok": ok,
        "gen_err": sum(r["gen_err"] for r in rows),
        "parse_fail": sum(r["parse_fail"] for r in rows),
        "retries_total": tot_retry,
        "retry_failed": sum(r["retry_failed"] for r in rows),
        "budget_stop": sum(r["budget_stop"] for r in rows),
        "avg_calls_per_q": round(avg_calls, 2),
        "avg_tpm_per_q": round(avg_tpm),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    out = Path(__file__).parent / f"measure_{label}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved: {out.name}")


if __name__ == "__main__":
    main()
