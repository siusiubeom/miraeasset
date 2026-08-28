# -*- coding: utf-8 -*-
"""processed/companies/*.md → processed/chunks/<listed_name>.jsonl

문서 단위(<!-- DOC --> 마커) → 헤더 경로 추적 → 문단/표행 단위로 목표 크기까지 패킹.
표 중간에서 청크가 끊기면 다음 청크에 표 헤더(첫 2줄)를 반복한다.
"""
import json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "processed" / "companies"
OUT_DIR = ROOT / "processed" / "chunks"

TARGET = 1500   # 청크 본문 목표 크기(문자)
HARD_MAX = 2400 # 단일 유닛이 넘으면 강제 분할
MIN_TAIL = 250  # 이보다 작은 꼬리 청크는 직전 청크에 흡수

DOC_RE = re.compile(
    r'<!-- DOC id=(?P<doc_id>\S+) group=(?P<group>\S+) subtype=(?P<subtype>\S+) '
    r'rcept_no=(?P<rcept_no>\S+) rcept_dt=(?P<rcept_dt>\S+) correction=(?P<corr>\S+) '
    # corp 는 공백을 포함할 수 있다(예: 'JYP Ent', 'LS ELECTRIC'). \S+ 로 받으면
    # 헤더 전체가 매칭 실패해 그 회사 청크가 0건이 된다.
    r'corp=(?P<corp>.+?) -->')
HDR_RE = re.compile(r'^(#{1,6}) (.*)$')


def split_units(body: str):
    """본문을 (문단 | 표 연속행 묶음) 유닛으로 분해. 표는 행 단위 유닛 + 헤더 정보."""
    units = []  # (text, table_header|None)
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            tbl = lines[i:j]
            hdr = "\n".join(tbl[:2]) if len(tbl) >= 2 and set(tbl[1].replace("|", "").strip()) <= set("-: ") else None
            # 표를 행 그룹으로: 헤더(있으면) 이후 행들을 개별 유닛으로
            start = 2 if hdr else 0
            if hdr:
                units.append((hdr, None))
            for k in range(start, len(tbl)):
                units.append((tbl[k], hdr))
            i = j
        else:
            j = i
            buf = []
            while j < len(lines) and not lines[j].lstrip().startswith("|"):
                buf.append(lines[j])
                j += 1
            for para in re.split(r"\n\s*\n", "\n".join(buf)):
                para = para.strip("\n")
                if para.strip():
                    units.append((para, None))
            i = j
    return units


def chunk_doc(meta: dict, body: str, report_nm: str):
    """헤더 경로를 유지하며 유닛을 청크로 패킹."""
    chunks = []
    path = []  # [(level, title)]
    cur = []          # 현재 청크의 유닛 텍스트들
    cur_len = 0
    cur_path = ""
    cur_tbl_hdr = None   # 현재 청크가 표 중간에서 시작해야 할 때 반복할 헤더
    pend_tbl_hdr = None  # 다음 청크 시작 시 반복할 표 헤더

    def breadcrumb():
        p = " > ".join(t for _, t in path if t)
        return f"[{meta['corp']} | {report_nm} | 접수 {meta['rcept_dt']} | {p}]" if p else \
               f"[{meta['corp']} | {report_nm} | 접수 {meta['rcept_dt']}]"

    def flush():
        nonlocal cur, cur_len, cur_tbl_hdr
        if not cur:
            return
        text = breadcrumb() + "\n" + ("\n".join(cur)).strip()
        chunks.append({"section_path": cur_path, "text": text})
        cur, cur_len, cur_tbl_hdr = [], 0, None

    for line in body.split("\n"):
        m = HDR_RE.match(line)
        if m:
            flush()
            lv, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= lv:
                path.pop()
            path.append((lv, title))
            cur_path = " > ".join(t for _, t in path)
            continue
        cur.append(line)
        cur_len += len(line) + 1
        if cur_len >= TARGET * 2:  # 러프 패킹: 세부 분할은 아래 재패킹에서
            pass

    flush()

    # 위는 헤더 단위 섹션. 이제 큰 섹션을 유닛 패킹으로 재분할.
    out = []
    for sec in chunks:
        txt = sec["text"]
        if len(txt) <= HARD_MAX:
            if txt.strip():
                out.append(sec)
            continue
        bc, _, body_txt = txt.partition("\n")
        units = split_units(body_txt)
        cur, cur_len, carry_hdr = [], 0, None
        packed = []
        for utext, uhdr in units:
            ulen = len(utext) + 1
            if cur_len + ulen > TARGET and cur:
                packed.append((cur, carry_hdr))
                cur, cur_len = [], 0
                carry_hdr = uhdr  # 표 중간이면 다음 청크에 헤더 반복
            cur.append(utext)
            cur_len += ulen
        if cur:
            packed.append((cur, carry_hdr))
        # 작은 꼬리 흡수
        if len(packed) >= 2 and sum(len(u) for u in packed[-1][0]) < MIN_TAIL:
            packed[-2][0].extend(packed[-1][0])
            packed.pop()
        for units_, hdr in packed:
            body_out = "\n".join(units_).strip()
            if hdr and not body_out.startswith(hdr):
                body_out = hdr + "\n" + body_out
            out.append({"section_path": sec["section_path"], "text": bc + "\n" + body_out})
    return out


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    manifest = {}
    for l in (ROOT / "corpus" / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        manifest[r["doc_id"]] = r
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_chunks = 0
    report = {}
    files = sorted(IN_DIR.glob("*.md"))
    if only:
        files = [f for f in files if f.stem == only]
    for f in files:
        text = f.read_text(encoding="utf-8")
        markers = list(DOC_RE.finditer(text))
        recs = []
        for i, m in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            meta = m.groupdict()
            mani = manifest.get(meta["doc_id"], {})
            report_nm = mani.get("report_nm", meta["subtype"])
            body = text[m.end():end]
            body = body.replace("=" * 80, "")
            for n, ch in enumerate(chunk_doc(meta, body, report_nm)):
                recs.append({
                    "chunk_id": f"{meta['doc_id']}#{n:04d}",
                    "corp": f.stem,
                    "corp_name": meta["corp"],
                    "doc_id": meta["doc_id"],
                    "group": meta["group"],
                    "subtype": meta["subtype"],
                    "report_nm": report_nm,
                    "rcept_no": meta["rcept_no"],
                    "rcept_dt": meta["rcept_dt"],
                    "correction": meta["corr"] == "Y",
                    "section_path": ch["section_path"],
                    "text": ch["text"],
                })
        with (OUT_DIR / f"{f.stem}.jsonl").open("w", encoding="utf-8") as fp:
            for r in recs:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[f.stem] = len(recs)
        total_chunks += len(recs)
        print(f"{f.stem}: {len(recs)} chunks", flush=True)
    (OUT_DIR / "_chunk_report.json").write_text(
        json.dumps({"total_chunks": total_chunks, "elapsed_sec": round(time.time() - t0, 1),
                    "per_company": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: {total_chunks} chunks in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
