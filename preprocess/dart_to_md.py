"""DART 공시 원문(XML/HTML) → 경량 마크다운 변환기.

원칙:
- 원문 텍스트는 수정하지 않는다 (용어 통일·요약 없음). 마크업만 제거.
- 제목(TITLE/COVER-TITLE/DOCUMENT-NAME), 문단(P), 표(TABLE)를 보존.
- 표의 rowspan/colspan은 값 반복으로 펼쳐 마크다운 표로 변환.
- 스타일·레이아웃 정보는 전부 버린다.

표준 라이브러리만 사용 (html.parser는 DART XML과 HTML 모두 관대하게 처리).
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

# 내용까지 통째로 버릴 태그 (닫는 태그가 있는 것만 — void 태그를 넣으면 스택이 안 풀림)
_SKIP_CONTENT = {"style", "script", "head", "summary", "colgroup"}
# 닫는 태그 없이 그 자체로 끝나는 태그 → 무시
_VOID_TAGS = {"meta", "link", "col", "img", "input", "pgbrk", "hr"}
# 셀로 취급할 태그 (DART는 TD 외에 TU/TE/TH를 씀)
_CELL_TAGS = {"td", "tu", "te", "th"}
_WS = re.compile(r"[ \t\xa0　]+")
_MANY_NL = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _cell_escape(text: str) -> str:
    return _clean(text.replace("\n", " ")).replace("|", "\\|")


class _Table:
    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self.cur: list[str] | None = None
        self.pending: dict[int, list] = {}  # col -> [남은 행수, 텍스트]
        self.col = 0

    def start_row(self) -> None:
        self.cur = []
        self.col = 0
        self._fill_pending()

    def _fill_pending(self) -> None:
        while self.col in self.pending:
            rem = self.pending[self.col]
            self.cur.append(rem[1])
            rem[0] -= 1
            if rem[0] <= 0:
                del self.pending[self.col]
            self.col += 1

    def add_cell(self, text: str, colspan: int, rowspan: int) -> None:
        if self.cur is None:
            self.start_row()
        self._fill_pending()
        # 짧은 라벨은 반복해 채우고(가독성·검색성), 긴 텍스트는 한 번만 쓴다
        rep = text if len(text) <= 60 else ""
        for i in range(colspan):
            cell = text if i == 0 else rep
            self.cur.append(cell)
            if rowspan > 1:
                self.pending[self.col] = [rowspan - 1, cell]
            self.col += 1
            self._fill_pending()

    def end_row(self) -> None:
        if self.cur is not None:
            self.rows.append(self.cur)
        self.cur = None

    def render(self) -> str:
        if self.cur is not None:
            self.end_row()
        rows = [r for r in self.rows if any(c.strip() for c in r)]
        if not rows:
            return ""
        ncols = max(len(r) for r in rows)
        lines = []
        for i, r in enumerate(rows):
            r = r + [""] * (ncols - len(r))
            lines.append("| " + " | ".join(r) + " |")
            if i == 0:
                lines.append("|" + "---|" * ncols)
        return "\n".join(lines)

    def render_inline(self) -> str:
        """중첩 표를 바깥 셀 안에 넣을 때 쓰는 압축 표현."""
        if self.cur is not None:
            self.end_row()
        rows = [r for r in self.rows if any(c.strip() for c in r)]
        return " ; ".join(" / ".join(c for c in r if c.strip()) for r in rows)


class DartToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip_stack: list[str] = []
        self.tables: list[_Table] = []
        self.cell_buf: list[str] | None = None
        self.cell_stack: list[list[str] | None] = []  # 중첩 표 진입 시 바깥 셀 버퍼 보관
        self.cell_span: tuple[int, int] = (1, 1)
        self.title_buf: list[str] | None = None
        self.title_level = 2
        self.text_buf: list[str] = []
        self.section_depth = 0

    # ---------- 유틸 ----------
    def _flush_text(self) -> None:
        t = _clean("".join(self.text_buf))
        self.text_buf = []
        if t:
            self.out.append(t + "\n")

    def _close_cell(self) -> None:
        if self.cell_buf is not None and self.tables:
            colspan, rowspan = self.cell_span
            self.tables[-1].add_cell(_cell_escape("".join(self.cell_buf)), colspan, rowspan)
        self.cell_buf = None

    # ---------- 파서 콜백 ----------
    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        if self.skip_stack:
            if tag in _SKIP_CONTENT:
                self.skip_stack.append(tag)
            return
        if tag in _SKIP_CONTENT:
            self.skip_stack.append(tag)
            return
        ad = {k.lower(): (v or "") for k, v in attrs}

        if re.fullmatch(r"section-\d+", tag):
            self.section_depth += 1
        elif tag == "table":
            self._flush_text()
            self._close_cell()  # 닫는 태그가 생략된 셀 방어
            if self.tables and self.cell_stack and self.cell_stack[-1] is not None:
                pass  # 중첩 표: 바깥 셀 버퍼는 cell_stack에 이미 보관됨
            self.cell_stack.append(self.cell_buf)
            self.cell_buf = None
            self.tables.append(_Table())
        elif tag == "tr" and self.tables:
            self._close_cell()
            self.tables[-1].start_row()
        elif tag in _CELL_TAGS and self.tables:
            self._close_cell()
            self.cell_buf = []
            try:
                colspan = max(1, int(ad.get("colspan", "1") or 1))
            except ValueError:
                colspan = 1
            try:
                rowspan = max(1, int(ad.get("rowspan", "1") or 1))
            except ValueError:
                rowspan = 1
            self.cell_span = (colspan, rowspan)
        elif tag in ("title", "cover-title", "document-name"):
            self._flush_text()
            self.title_buf = []
            self.title_level = 1 if tag in ("cover-title", "document-name") else min(2 + self.section_depth, 6)
        elif tag in ("p", "div", "li"):
            if self.cell_buf is not None:
                self.cell_buf.append(" ")
            elif self.title_buf is None:
                self._flush_text()
        elif tag == "br":
            if self.cell_buf is not None:
                self.cell_buf.append(" ")
            else:
                self.text_buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip_stack:
            if tag == self.skip_stack[-1]:
                self.skip_stack.pop()
            return
        if re.fullmatch(r"section-\d+", tag):
            self.section_depth = max(0, self.section_depth - 1)
        elif tag == "table" and self.tables:
            self._close_cell()
            t = self.tables.pop()
            outer_cell = self.cell_stack.pop() if self.cell_stack else None
            if self.tables and outer_cell is not None:
                # 중첩 표 → 바깥 셀 안에 압축 문자열로
                outer_cell.append(" [" + t.render_inline() + "] ")
                self.cell_buf = outer_cell
            else:
                self.cell_buf = outer_cell
                md = t.render()
                if md:
                    self.out.append("\n" + md + "\n")
        elif tag == "tr" and self.tables:
            self._close_cell()
            self.tables[-1].end_row()
        elif tag in _CELL_TAGS:
            self._close_cell()
        elif tag in ("title", "cover-title", "document-name"):
            if self.title_buf is not None:
                t = _clean("".join(self.title_buf))
                self.title_buf = None
                if t:
                    self.out.append("\n" + "#" * self.title_level + " " + t + "\n")
        elif tag == "p":
            if self.cell_buf is None and self.title_buf is None:
                self._flush_text()

    def handle_data(self, data: str) -> None:
        if self.skip_stack or not data:
            return
        if self.cell_buf is not None:
            self.cell_buf.append(data)
        elif self.title_buf is not None:
            self.title_buf.append(data)
        else:
            self.text_buf.append(data)

    def result(self) -> str:
        self._flush_text()
        while self.tables:  # 비정상 종료 방어
            t = self.tables.pop()
            md = t.render()
            if md:
                self.out.append("\n" + md + "\n")
        text = "".join(self.out)
        return _MANY_NL.sub("\n\n", text).strip() + "\n"


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def convert_file(path: Path) -> str:
    parser = DartToMarkdown()
    parser.feed(read_text(path))
    parser.close()
    return parser.result()


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        p = Path(arg)
        md = convert_file(p)
        out = p.with_suffix(".md")
        out.write_text(md, encoding="utf-8")
        print(f"{p} ({p.stat().st_size:,}B) -> {out} ({len(md.encode('utf-8')):,}B)")
