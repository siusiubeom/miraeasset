"""공시의 산출근거 블록을 출처와 함께 보존한다. 질문 의도 판정 없음."""
import re

MARKER = re.compile(r"산출\s*근거|【[^】]*(?:금액\s*한도|산식)[^】]*】|[①②③]")


def extract_blocks(hits):
    blocks, seen = [], set()
    for rec, _ in hits:
        lines = (rec.get("text") or "").splitlines()
        selected = set()
        for i, line in enumerate(lines):
            if not MARKER.search(line):
                continue
            # 표 행과 연속된 산식 문단을 함께 읽고 다음 절에서 멈춘다.
            start = i
            while start > 0 and lines[start - 1].strip() and not lines[start - 1].startswith("#"):
                start -= 1
            end = i + 1
            while end < len(lines) and lines[end].strip() and not lines[end].startswith("#"):
                end += 1
            selected.update(range(start, end))
        if selected:
            body = "\n".join(lines[i] for i in sorted(selected))
            key = (rec["rcept_no"], body)
            if key not in seen:
                seen.add(key)
                blocks.append(f"접수번호 {rec['rcept_no']} | {rec.get('section_path', '')}\n{body}")
    return "\n\n".join(blocks)
