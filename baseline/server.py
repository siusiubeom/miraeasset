# -*- coding: utf-8 -*-
"""평가용 /answer API 서버 (표준 라이브러리만 사용).

  GET /answer?question_id=Q-001&question=...
  → 200 JSON {question_id, question, retrieved_context, think_trace, answer} (모두 string)

운영 원칙 (평가 규격 반영):
- 어떤 내부 오류에도 5xx를 내지 않고 200 + 성실한 JSON으로 응답
- 문항당 300초 타임아웃 → 내부 285초 가드, 초과 시 그 시점까지의 부분 응답 반환
- 재시도(동일 question_id 재수신) 대비 응답 캐시(멱등)
- 평가 호출은 순차 1건씩이므로 단일 프로세스로 충분

실행:  py server.py [port]   (기본 8080; 운영 시 80)
"""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from answerer import answer_question, get_retriever

GUARD_SEC = 285
_cache = {}          # question_id → response dict
_cache_lock = threading.Lock()


def safe_answer(question_id: str, question: str) -> dict:
    """타임아웃·예외에도 항상 규격 JSON을 돌려준다."""
    result = {}

    def work():
        try:
            result["resp"] = answer_question(question_id, question)
        except Exception as e:
            result["resp"] = {
                "question_id": question_id, "question": question,
                "retrieved_context": "", "think_trace": f"internal error: {type(e).__name__}: {e}",
                "answer": "일시적인 내부 오류로 답변을 생성하지 못했습니다. 제공된 공시에서 확인되지 않습니다.",
            }

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(GUARD_SEC)
    if "resp" not in result:
        return {
            "question_id": question_id, "question": question,
            "retrieved_context": "", "think_trace": f"timeout: {GUARD_SEC}s 내 파이프라인 미완료",
            "answer": "제한 시간 내에 답변을 완성하지 못했습니다. 제공된 공시에서 확인되지 않습니다.",
        }
    return result["resp"]


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            self._send_json({"status": "ok"})
            return
        if url.path != "/answer":
            self._send_json({"error": "use GET /answer?question_id=..&question=.."}, status=404)
            return
        try:  # UTF-8 우선, 윈도우 클라이언트의 CP949 인코딩 질의도 수용
            qs = parse_qs(url.query, encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            qs = parse_qs(url.query, encoding="cp949", errors="replace")
        qid = (qs.get("question_id") or [""])[0]
        question = (qs.get("question") or [""])[0]
        if not question:
            self._send_json({"question_id": qid, "question": "", "retrieved_context": "",
                             "think_trace": "missing 'question' parameter", "answer": ""})
            return
        with _cache_lock:
            cached = _cache.get(qid) if qid else None
        if cached and cached["question"] == question:
            self.log_message("cache hit for %s", qid)
            self._send_json(cached)
            return
        t0 = time.time()
        resp = safe_answer(qid, question)
        self.log_message("answered %s in %.1fs", qid, time.time() - t0)
        if qid:
            with _cache_lock:
                _cache[qid] = resp
        self._send_json(resp)

    def log_message(self, fmt, *args):  # 기본 stderr 로그 유지하되 타임스탬프 포함
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    get_retriever()  # 유니버스/정정링크 선로딩 (인덱스는 질의 시 lazy)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"listening on :{port}  (GET /answer, /health)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
