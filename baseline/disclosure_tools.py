"""공시 서식에서 확인된 조건과 표를 계산한다."""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import re

import aggregate_tools as AGG


@dataclass
class ToolResult:
    subject: str
    answer: str
    sources: list
    result: str
    limitation: str
    exclusions: list = field(default_factory=list)
    basis: str = ''


def compact(text):
    return re.sub(r"\s+", "", text)


def number(text):
    text = text.strip().replace(",", "").replace("−", "-")
    if text == "-":
        return Decimal(0)
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def fmt(value):
    return f"{value:,.0f}" if value == value.to_integral_value() else f"{value:,f}"


def date_values(text):
    out = []
    pattern = r"(?P<y>20\d{2})\s*(?:년|[-./])\s*(?P<m>\d{1,2})\s*(?:(?P<kr>월)|[-./])\s*(?P<d>\d{1,2})(?(kr)\s*일|(?!\d|,\d))"
    for match in re.finditer(pattern, text):
        y, m, d = match['y'], match['m'], match['d']
        try:
            value = date(int(y), int(m), int(d))
        except ValueError:
            continue
        if value not in out:
            out.append(value)
    return out


def rows(text):
    for line in text.splitlines():
        if line.strip().startswith("|"):
            yield [c.strip() for c in line.strip().strip("|").split("|")], line


def unique_field(text, labels, dates=False):
    values, evidence = set(), []
    for cells, line in rows(text):
        if all(label in compact(line) for label in labels):
            vals = date_values(line) if dates else [number(c) for c in cells]
            vals = [v for v in vals if v is not None]
            if vals:
                values.update(vals)
                evidence.append(line)
    return (next(iter(values)) if len(values) == 1 else None), evidence


def document(corp, meta):
    records = AGG.chunks_by_rcept(corp).get(meta["rcept_no"], [])
    text = "\n".join(dict.fromkeys(r.get("text", "") for r in records))
    source = dict(records[0]) if records else dict(meta, corp=corp)
    source.update(text=text, corp=corp, group=meta.get("doc_group", source.get("group", "major")),
                  rcept_no=meta['rcept_no'], rcept_dt=meta['rcept_dt'], report_nm=meta['report_nm'])
    return source


def cite(source):
    return f"{source['corp']} {source['report_nm']}, 접수 {source['rcept_dt']}, 접수번호 {source['rcept_no']}"


def cb_terms(question, corp):
    q = compact(question)
    if not ("CB" in question or "전환사채" in q):
        return None
    wants_dates = "전환청구기간" in q and "날짜" in q
    wants_shares = "전환가능주식수" in q and "전환가액" in q
    if not (wants_dates or wants_shares):
        return None
    series = re.search(r"제\s*(\d+)\s*회", question)
    candidates = []
    for m in AGG.docs_for(corp, {"report": "전환사채권발행결정"}):
        source = document(corp, m)
        issue = re.search(r"회차\s*\|\s*(\d+)\s*\|", source["text"])
        if issue and (not series or issue[1] == series[1]):
            candidates.append((issue[1], source))
    if not candidates or len({s for s, _ in candidates}) != 1:
        return None
    # 같은 회차의 최초 조건과 마지막으로 관측된 발행결정 조건을 구분한다.
    source = candidates[0 if wants_shares else -1][1]
    text = source["text"]
    fields = {}
    evidence = []
    specs = {"start": (["전환청구기간", "시작일"], True),
             "end": (["전환청구기간", "종료일"], True),
             "maturity": (["사채만기일"], True),
             "principal": (["2.사채의권면", "총액(원)"], False),
             "price": (["전환가액(원/주)"], False),
             "floor": (["최저조정가액(원)"], False)}
    for name, (labels, is_date) in specs.items():
        fields[name], lines = unique_field(text, labels, dates=is_date)
        evidence.extend(lines)
    limitation = "발행결정에 기재된 계약 조건이며 기준일의 실제 미상환 잔액, 후속 조건 변경 및 실제 전환 가능 여부까지 확인한 것은 아니다"
    if wants_dates:
        asked = date_values(question)
        if len(asked) != 1 or any(fields[k] is None for k in ("start", "end", "maturity")):
            return None
        start, end, maturity = (fields[k] for k in ("start", "end", "maturity"))
        if start > end or end > maturity:
            return None
        status = "포함됩니다" if start <= asked[0] <= end else "포함되지 않습니다"
        result = f"{start} ≤ {asked[0]} ≤ {end}" if start <= asked[0] <= end else f"{asked[0]}은 {start}~{end} 밖"
        answer = (f"제{candidates[0][0]}회 CB의 전환청구 시작일은 {start}, 종료일은 {end}, 사채 만기일은 {maturity}입니다. "
                  f"{asked[0]}은 공시상 전환청구기간에 {status}. 이 판정은 날짜 조건만 비교한 결과입니다. "
                  "전환청구 종료일과 사채 만기일은 다릅니다.")
    else:
        principal, price, floor = (fields[k] for k in ("principal", "price", "floor"))
        if any(v is None or v <= 0 for v in (principal, price, floor)) or floor > price:
            return None
        clauses = [line for line in text.splitlines() if "단주" in line and "현금" in line]
        if not clauses:
            return None
        evidence.extend(clauses)
        initial = (principal / price).to_integral_value(rounding=ROUND_FLOOR)
        maximum = (principal / floor).to_integral_value(rounding=ROUND_FLOOR)
        remainder = principal - maximum * floor
        ratio = floor / price * 100
        result = f"최초 {fmt(initial)}주; 하한 적용 {fmt(maximum)}주; 증가 {fmt(maximum-initial)}주"
        answer = (f"제{candidates[0][0]}회 CB의 최초 발행총액은 {fmt(principal)}원, 전환가액은 {fmt(price)}원/주입니다.\n"
                  f"최초 전환가능주식수 = {fmt(principal)} / {fmt(price)} = {fmt(initial)}주.\n"
                  f"계약상 최저 조정가액은 {fmt(floor)}원/주로 최초 가액의 {fmt(ratio)}%입니다.\n"
                  f"하한 적용 전환가능주식수 = floor({fmt(principal)} / {fmt(floor)}) = {fmt(maximum)}주, "
                  f"최초 대비 {fmt(maximum-initial)}주 증가합니다.\n"
                  f"단주에 해당하는 원금 잔액은 {fmt(remainder)}원입니다. 원문은 1주 미만 단주를 전환하지 않고 현금 지급한다고 규정합니다. "
                  "발행총액 전액이 남아 있고 해당 하한이 적용된다는 가정의 계산이며 실제 리픽싱이나 전환 발생을 뜻하지 않습니다.")
    source["text"] = "\n".join(dict.fromkeys(evidence))
    return ToolResult(f"{corp} 전환 조건", answer + "\n\n" + limitation + ".\n근거: " + cite(source),
                      [source], result, limitation)


def correction_example(question, corp):
    q = compact(question)
    if not all(k in q for k in ("정정", "계약금액", "계약기간", "발생", "사례")):
        return None
    for meta in AGG.docs_for(corp, {"report": "공급계약", "correction": True}):
        years = re.findall(r"(20\d{2})년", question)
        if years and meta["rcept_dt"][:4] not in years:
            continue
        source = document(corp, meta)
        changes = AGG.correction_changes(source["text"])
        selected = {c["category"] for c in changes if c["category"]}
        if selected != {"계약금액", "계약기간"}:
            continue
        if not any(c['category'] == '계약금액' and number(c['before']) not in (None, Decimal(0))
                   and number(c['after']) not in (None, Decimal(0)) for c in changes):
            continue
        if not any(c['category'] == '계약기간' and date_values(c['before']) and date_values(c['after']) for c in changes):
            continue
        answer = [f"{cite(source)}에서는 계약금액과 계약기간이 함께 변경됐습니다."]
        answer += [f"- {c['field']}: {c['before']} → {c['after']}" for c in changes if c["category"]]
        answer.append("문서 수로는 1건입니다. 금액·기간의 사유 발생 횟수로는 계약금액 변경 1회와 계약기간 변경 1회, 총 2회입니다. "
                      "같은 문서가 두 범주에 각각 포함되며, 변경된 표 행의 수를 그대로 사유 횟수로 더하지 않습니다.")
        return ToolResult(f"{corp} 정정 집계 단위", "\n".join(answer), [source],
                          "동일 문서 1건에서 계약금액과 계약기간의 두 범주를 확인해 발생 횟수를 2회로 집계했다",
                          "이 사례의 두 범주 발생 횟수이며 회사 전체의 정정사유 횟수를 집계한 것은 아니다")
    return None


def treasury_formula(question, corp):
    q = compact(question)
    quota = "취득금액한도" in q and "취득원가" in q
    daily = "1일매수" in q and "한도" in q
    if not (quota or daily):
        return None
    candidates = []
    years = re.findall(r"(20\d{2})년", question)
    for meta in AGG.docs_for(corp, {"report": "자기주식취득결정"}):
        if years and meta["rcept_dt"][:4] not in years:
            continue
        source = document(corp, meta)
        text = source["text"]
        if quota:
            if not all(term in compact(text) for term in ("한도(1-2-3-4-5+6)", "2.직전사업연도말이후자기주식취득금액", "6.직전사업연도말이후자기주식처분시처분주식의취득원가")):
                continue
            lines = [line for line in text.splitlines() if "직전" in line or "한도(1-" in line]
            source["text"] = "\n".join(lines)
            answer = ("공시의 자기주식 취득금액 한도 산식은 1−2−3−4−5+6입니다. "
                      "2번은 직전 사업연도말 이후 자기주식 취득금액으로, 이미 사용한 취득 여력을 차감하는 항목입니다. "
                      "6번은 이후 처분한 자기주식의 취득원가로, 처분으로 더 이상 보유하지 않는 주식에 대응하는 과거 취득원가만큼 한도를 복원합니다. "
                      "따라서 가산하는 것은 처분대금이나 처분이익이 아닙니다. "
                      + ("공시는 취득원가에 이동평균법을 적용한다고 기재합니다. " if '이동평균법' in text else '') +
                      "이는 공시 한도 산식의 항목과 부호를 설명한 것이며, 모든 자기주식 거래의 법적 한도를 별도로 산정한 것은 아닙니다.")
            return ToolResult(f"{corp} 취득 한도", answer + "\n근거: " + cite(source), [source],
                              "산식의 차감 항목 2와 가산 항목 6을 대조해 처분대금과 취득원가를 구분했다",
                              "회계 분개 전체가 아니라 해당 공시 한도 산식의 항목 의미를 설명했다")
        plain = "\n".join(line for line in text.splitlines() if not line.startswith("|"))
        match = re.search(r"①\s*취득 신고.*?③\s*발행주식총수.*?\n[^\n]+", plain, re.S)
        if not match or "①과 ② 중 많은 수량" not in plain or "③ 중 적은 수량" not in plain:
            continue
        block = match[0]
        terms = re.split(r"[①②③]", block)[1:]
        if len(terms) != 3:
            continue
        values = {}
        for stock in ("보통주", "우선주"):
            found = [re.search(stock + r"\s*([\d,]+)주", term) for term in terms]
            if all(found):
                values[stock] = [Decimal(m[1].replace(",", "")) for m in found]
        if len(values) != 2:
            continue
        candidates.append((source, values, block))
    if not candidates:
        return None
    if "다른" in q:
        differing = [item for item in candidates if len({a > b for a, b, c in item[1].values()}) > 1]
        if not differing:
            return None
        candidates = differing
    source, values, block = candidates[-1]
    lines = [f"코퍼스에서 질문의 기준 차이를 확인할 수 있는 사례({source['rcept_dt']})를 기준으로 설명합니다.",
             "1일 매수한도 = min(max(A, B), C)입니다. A는 취득신고수량의 10%, B는 이사회 결의일 전 1개월 일평균거래량의 25%, C는 발행주식총수의 1%입니다."]
    for stock, (a, b, c) in values.items():
        high = max(a, b)
        bound = min(high, c)
        winner = "A" if a > b else "B" if b > a else "A와 B(동일)"
        lines.append(f"{stock}: A={fmt(a)}주, B={fmt(b)}주, C={fmt(c)}주. max(A,B)는 {winner}의 {fmt(high)}주이며, "
                     f"C와 비교한 최종 한도는 {fmt(bound)}주입니다.")
    lines.append("일반적인 거래량 차이를 가정한 것이 아니라, 각 주식 종류의 실제 신고수량 기준과 거래량 기준을 비교한 결과입니다.")
    lines.append("근거: " + cite(source))
    source["text"] = block + "\n" + "\n".join(line for line in source["text"].splitlines() if "산출근거" in line or "1일 매수 주문" in line)
    return ToolResult(f"{corp} 일일 매수 한도", "\n".join(lines), [source],
                      "; ".join(f"{s} min(max({fmt(a)},{fmt(b)}),{fmt(c)})={fmt(min(max(a,b),c))}주" for s,(a,b,c) in values.items()),
                      "확인한 공시의 산출근거에 한정하며 다른 취득결정의 한도와 혼합하지 않았다")


def equity_rollforward(question, corp):
    q = compact(question)
    if not ("지분법" in q and "배당" in q):
        return None
    separate = "별도재무제표" in q or "별도기준" in q
    years = re.findall(r"20\d{2}", question)
    candidates = []
    for meta in AGG.docs_for(corp, {"report": "사업보고서"}):
        report_year = re.search(r"\((20\d{2})\.", meta['report_nm'])
        if not report_year or (years and report_year[1] not in years):
            continue
        candidates.append((meta, int(report_year[1])))
    for meta, report_year in reversed(candidates):
        records = AGG.chunks_by_rcept(corp).get(meta['rcept_no'], [])
        sections = {}
        for r in records:
            path = r.get('section_path', '')
            if '관계기업' in compact(path) and separate == ('연결' not in path):
                sections.setdefault(path, []).append(r)
        found = []
        for path, chunks in sections.items():
            text = '\n'.join(r['text'] for r in chunks)
            headers, period, unit = None, None, None
            for line in text.splitlines():
                normalized = compact(line)
                if re.search(r'제\d+\(?당\)?기', normalized):
                    period = report_year
                elif re.search(r'제\d+\(?전\)?기', normalized):
                    period = report_year - 1
                u = re.search(r'단위[:：](백만원|천원|억원|원)', normalized)
                if u:
                    unit = u[1]
                if not line.strip().startswith('|'):
                    continue
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                ns = [compact(c) for c in cells]
                if all(k in ns for k in ('기초', '기말', '지분법손익', '배당금수령')):
                    headers = ns
                    continue
                if ns[0] in ('회사명', '구분'):
                    headers = None
                    continue
                if not headers or len(cells) != len(headers) or not period or not unit:
                    continue
                if years and str(period) not in years:
                    continue
                vals = [number(c) for c in cells[1:]]
                if not vals or any(v is None for v in vals) or headers.index('기초') != 1 or headers.index('기말') != len(cells)-1:
                    continue
                if sum(vals[:-1]) != vals[-1]:
                    continue
                name = re.sub(r'㈜|\(주\)|주식회사', '', cells[0]).strip()
                if not name or name in ('합계','소계'):
                    continue
                found.append((name, period, unit, headers, vals, line, path, text, name in question))
        matches = [r for r in found if r[-1]]
        if not matches:
            # 대상 관계기업을 지정하지 않은 경우 장부가액 큰 사례를 명시해 제공한다.
            if '주요' not in q:
                continue
            matches = sorted(found, key=lambda r: r[4][-1], reverse=True)[:1]
        if not matches:
            continue
        if len({(r[0],r[1],r[2],tuple(r[4])) for r in matches}) != 1:
            continue
        name, period, unit, headers, vals, line, path, text, _ = matches[0]
        dividends, earnings = vals[headers.index('배당금수령')-1], vals[headers.index('지분법손익')-1]
        if dividends > 0:
            continue
        equation = fmt(vals[0]) + ''.join((' − ' if v < 0 else ' + ') + fmt(abs(v)) for v in vals[1:-1]) + ' = ' + fmt(vals[-1]) + unit
        entries = '; '.join(f'{h} {fmt(v)}{unit}' for h,v in zip(headers[1:],vals))
        source = document(corp, meta)
        source.update(section_path=path, text=text)
        answer = (f"{period}년 {'별도' if separate else '연결'} 관계기업투자 변동표의 {name} 사례입니다. "
                  f"지분법손익은 {fmt(earnings)}{unit}, 배당금 수령액은 {fmt(abs(dividends))}{unit}입니다.\n"
                  f"변동내역: {entries}.\n검산: {equation}.\n"
                  "이 표에서 지분법손익은 투자 장부가액에 반영되고 배당금 수령은 차감 항목입니다. "
                  "배당 수령액을 지분법이익에 추가 수익으로 다시 더하면 같은 이익을 중복 반영하게 됩니다. "
                  "배당수령과 지분법손익은 서로 다른 변동 항목으로 구분해야 합니다.\n근거: " + cite(source))
        return ToolResult(f'{corp} 관계기업 변동', answer, [source], '기초와 각 변동항목의 합을 계산한 결과 기말 장부가액과 일치했다: '+equation,
                          f'{period}년 {name}의 해당 투자 변동표에 한정한 검산이다')
    return None


def segment_reconciliation(question, corp):
    q = compact(question)
    if not all(k in q for k in ('영업이익','비중','대조')):
        return None
    years = re.findall(r'20\d{2}', question)
    if len(set(years)) != 1:
        return None
    for meta in reversed(AGG.docs_for(corp, {'report':'사업보고서'})):
        if f'({years[0]}.' not in meta['report_nm']:
            continue
        records = AGG.chunks_by_rcept(corp).get(meta['rcept_no'], [])
        sections = {}
        for r in records:
            if '재무상태및영업실적' in compact(r.get('section_path','')):
                sections.setdefault(r['section_path'],[]).append(r)
        for path, recs in sections.items():
            text = '\n'.join(r['text'] for r in recs)
            fiscal = re.search(r'제(\d+)기\(당기\)', compact(text))
            if not fiscal:
                continue
            current_col, unit, items = None, None, []
            for line in text.splitlines():
                u = re.search(r'단위[:：](백만원|천원|억원|원)', compact(line))
                if u:
                    unit = u[1]
                if not line.strip().startswith('|'):
                    continue
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                if len(cells) < 4:
                    continue
                if compact(cells[0]) == '구분' and compact(cells[1]) == '부문' and any('제' in c and '기' in c for c in cells[2:]):
                    current_col = next((i for i,c in enumerate(cells) if compact(c) == f'제{fiscal[1]}기'), None)
                    continue
                if current_col is None or len(cells) <= current_col + 1 or compact(cells[0]) != '영업이익':
                    continue
                value = number(cells[current_col])
                pct = number(cells[current_col+1].removesuffix('%')) if cells[current_col+1].endswith('%') else None
                if value is not None and pct is not None and unit:
                    items.append((cells[1], value, pct, unit, line))
            wholes = [r for r in items if '전사' in r[0] or '기업전체' in compact(r[0])]
            parts = [r for r in items if compact(r[0]).removesuffix('부문') in q and r not in wholes]
            if len(wholes) != 1 or len(parts) != 1 or parts[0][3] != wholes[0][3] or wholes[0][1] == 0:
                continue
            part, whole = parts[0], wholes[0]
            calc = (part[1] / whole[1] * 100).quantize(Decimal('0.01'))
            precision = Decimal(1).scaleb(part[2].as_tuple().exponent)
            agrees = calc.quantize(precision) == part[2]
            answer = (f"{years[0]}년 {part[0]} 영업이익은 {fmt(part[1])}{part[3]}, 전사 영업이익은 {fmt(whole[1])}{whole[3]}입니다.\n"
                      f"비중 = {fmt(part[1])} / {fmt(whole[1])} × 100 = {calc}%. "
                      f"공시에 기재된 비중은 {part[2]}%이며, 표시 자릿수로 반올림하면 {'일치합니다' if agrees else '일치하지 않습니다'}. ")
            if part[3] == '백만원':
                answer += f"{fmt(part[1])}백만원은 {fmt(part[1]/100)}억원입니다. 금액의 단위를 양쪽에 동일하게 환산해도 비중은 같습니다."
            source = document(corp,meta)
            source.update(section_path=path,text=text)
            return ToolResult(f'{corp} 부문 비중 대조',answer+'\n근거: '+cite(source),[source],
                              f'계산 비중 {calc}%와 공시 표시 {part[2]}%를 대조한 결과 '+('반올림 정밀도에서 일치했다' if agrees else '불일치했다'),
                              '동일 보고서의 당기 부문 및 전사 영업이익을 사용했다')
    return None


def run(question, corp):
    import comparative_reports
    import event_records
    result = event_records.run(question, corp)
    if result:
        return result
    result = comparative_reports.run(question, corp)
    if result:
        return result
    for tool in (cb_terms, correction_example, treasury_formula, equity_rollforward, segment_reconciliation):
        result = tool(question, corp)
        if result:
            return result
    return None
