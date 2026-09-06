"""실행된 조회·계산과 확인 한계로 판단 요약을 구성한다."""
import re


def clean_sentence(text):
    text = re.sub(r"\s+", " ", text).strip(" -.")
    return text + "." if text else ""


def render(state, trace):
    hits = state.get("evidence_hits") or []
    docs = list({r["rcept_no"]: r for r, _ in hits}.values())
    docs += [r for r in state.get("audit_docs", []) if r["rcept_no"] not in {d["rcept_no"] for d in docs}]
    subject = state.get("audit_subject")
    if not subject and docs:
        first = docs[0]
        topic = (first.get("section_path") or first.get("report_nm") or "공시").split(" > ")[-1]
        topic = re.sub(r"^\d+[.-]?\s*", "", topic)
        subject = f"{first.get('corp', '')} {topic}"
    title = re.sub(r"\s+", " ", subject or "조회 범위 확인").strip()[:24].rstrip(". ")
    lines = []
    if state.get('audit_basis'):
        lines.append(clean_sentence(state['audit_basis']))
    if docs:
        labels = []
        for r in docs[:2]:
            dt = r.get('rcept_dt', '')
            filed = f"접수 {dt[:4]}-{dt[4:6]}-{dt[6:8]}, " if re.fullmatch(r'\d{8}',dt) else ''
            section = r.get('section_path','').split(' > ')[-1]
            where = f', {section}' if section and section != r.get('report_nm') else ''
            labels.append(f"{r.get('report_nm', '공시')}{where} ({filed}접수번호 {r['rcept_no']})")
        refs = "; ".join(labels)
        lines.append(f"근거로 제공한 {len(docs)}개 문서 중 {refs}를 조회했다.")
    result = state.get("audit_result")
    if not result:
        calcs = [s.removeprefix("[4S+] ") for s in trace if s.startswith("[4S+] ")]
        result = "; ".join(c for c in calcs[:2] if len(c) <= 260)
        if result:
            result = "표에서 계산한 결과는 " + result
    if result:
        lines.append(clean_sentence(result))
    exclusions = state.get("audit_exclusions") or []
    if exclusions:
        lines.append(clean_sentence("; ".join(exclusions[:2])))
    limitation = state.get("audit_limit")
    if not limitation:
        correction = [s.removeprefix("- ") for s in state.get("judgment_lines", []) if "정정" in s]
        limitation = correction[0] if correction else "이 요약의 범위는 제공된 근거와 실제 실행 기록에 한정된다"
    lines.append(clean_sentence(limitation))
    if not docs and not result:
        reason = next((s for s in trace if "거절 판정:" in s), None)
        if reason:
            lines.insert(0, clean_sentence(reason.split("거절 판정:", 1)[1] + " 사유로 수치 답변을 생성하지 않았다"))
    return title + "\n" + "\n".join(lines)
