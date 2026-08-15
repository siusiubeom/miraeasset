# -*- coding: utf-8 -*-
"""CLOVA Studio v3 chat-completions 규격을 흉내내는 로컬 목 서버 (통합 검증용).

실제 응답 스키마(status/result.message.content, 401 코드 40104 등)를 재현한다.
생성 내용은 '근거 컨텍스트에서 질문 키워드 주변을 인용'하는 규칙 기반 — 파이프라인
배관(요청 포맷, 인증 헤더, 타임아웃, 파싱, 폴백)을 검증하는 용도이지 품질 평가용이 아니다.

사용: py mock_clova.py [port=8099]   (서버측: CLOVA_ENDPOINT=http://127.0.0.1:8099/v3/chat-completions/HCX-005)
"""
import json, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VALID_KEY_PREFIX = "nv-"


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer " + VALID_KEY_PREFIX):
            self._json({"status": {"code": "40104", "message": "Invalid Key - Please use new API Key that starts with 'nv-*'."}}, 401)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
            msgs = req["messages"]
            assert isinstance(msgs, list) and all(m["role"] in ("system", "user", "assistant") for m in msgs)
        except Exception as e:
            self._json({"status": {"code": "40000", "message": f"Bad Request: {e}"}}, 400)
            return
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        q = user.rsplit("[질문]", 1)[-1].strip()
        ctx = user.split("[질문]")[0]
        # 규칙 기반 '생성': 질문 키워드가 나오는 근거 블록의 출처 라벨 + 해당 줄 인용
        keywords = [w for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", q) if w not in ("얼마인가", "어디인가", "누구인가", "무엇인가", "정리해줘", "설명해줘")]
        blocks = re.split(r"===== \[근거 (\d+)\] ", ctx)[1:]
        picked = []
        for i in range(0, len(blocks) - 1, 2):
            head, body = blocks[i], blocks[i + 1]
            src = body.split("=====")[0].strip()
            lines = [l for l in body.split("\n") if sum(k in l for k in keywords) >= 1 and any(c.isdigit() for c in l)]
            if lines and len(picked) < 2:
                picked.append((src, lines[:3]))
        if picked:
            parts = []
            for src, lines in picked:
                parts.append(f"근거 공시({src})에 따르면:\n" + "\n".join("  " + l.strip() for l in lines))
            content = "\n".join(parts) + "\n(모의 생성 응답 — 배관 검증용)"
        else:
            content = "제공된 공시에서 확인되지 않습니다. (모의 생성 응답)"
        self._json({"status": {"code": "20000", "message": "OK"},
                    "result": {"message": {"role": "assistant", "content": content},
                               "stopReason": "stop", "inputLength": len(user), "outputLength": len(content)}})

    def log_message(self, fmt, *args):
        sys.stderr.write("[mock-clova] " + fmt % args + "\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"mock CLOVA on :{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
