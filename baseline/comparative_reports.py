"""동일 보고서의 기간·기준·표 구분을 보존해 비교한다."""
from dataclasses import dataclass
from decimal import Decimal
import re

import aggregate_tools as AGG
from disclosure_tools import ToolResult, compact, number, fmt, cite


@dataclass(frozen=True)
class Observation:
    label: str
    period: int
    unit: str
    value: Decimal
    block: str
    line: str


def annual(corp, year=None):
    docs = AGG.docs_for(corp, {'report': '사업보고서'})
    docs = [m for m in docs if re.search(r'\(20\d{2}\.12\)', m['report_nm'])]
    if year:
        docs = [m for m in docs if f'({year}.12)' in m['report_nm']]
    return docs[-1] if docs else None


def sections(corp, meta, predicate, financial_only=True):
    out = {}
    for r in AGG.chunks_by_rcept(corp).get(meta['rcept_no'], []):
        path = r.get('section_path', '')
        if (not financial_only or 'III. 재무에 관한 사항' in path) and predicate(compact(path)):
            out.setdefault(path, []).append(r)
    return [dict(recs[0], corp=corp, report_nm=meta['report_nm'], rcept_dt=meta['rcept_dt'],
                 text='\n'.join(dict.fromkeys(r['text'] for r in recs))) for recs in out.values()]


def label(text):
    return re.sub(r'\(주\d[^)]*\)', '', compact(text))


def observations(text, report_year):
    """법정 기수 열 또는 명시된 당기/전기 블록만 해석한다. 알 수 없는 열은 버린다."""
    fiscal = {m[1]: int(m[2]) for m in re.finditer(r'제\s*(\d+)\s*기\s*(20\d{2})[.]\d{2}[.]\d{2}', text)}
    period, columns, unit, block = None, {}, None, ''
    out = []
    for line in text.splitlines():
        if not line.strip().startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        ns = [compact(c) for c in cells]
        if not cells or all(re.fullmatch(r'[-:]*', c) for c in ns):
            continue
        u = re.search(r'단위[:：](백만원|천원|억원|원|주)', compact(line))
        if u and len(cells) <= 2 and not any(number(c) is not None for c in cells[1:]):
            unit = u[1]
        if ns[0] in ('당기','전기','전전기'):
            period = report_year - ('당기','전기','전전기').index(ns[0])
            columns = {}
            continue
        mapped = {}
        for i,c in enumerate(ns):
            m = re.fullmatch(r'제(\d+)기', c)
            if m and m[1] in fiscal:
                mapped[i] = fiscal[m[1]]
            explicit = re.fullmatch(r'(?:제?\d+기)?(20\d{2})년연간',c)
            if explicit:
                mapped[i] = int(explicit[1])
        if mapped:
            columns, period = mapped, None
            continue
        vals = [(i,number(c)) for i,c in enumerate(cells[1:],1) if number(c) is not None]
        if not vals:
            if len(set(ns)) == 1 and ns[0] and not ns[0].startswith(('제','(')) and len(ns[0]) < 100:
                block = ns[0]
                period, columns = None, {}
            continue
        effective_unit = u[1] if u else ('주' if '(주)' in ns[0] and '주식수' in ns[0] else unit)
        if not effective_unit:
            continue
        for i,v in vals:
            y = columns.get(i) if columns else period if len(vals)==1 else None
            if y is not None:
                out.append(Observation(label(cells[0]),y,effective_unit,v,block,line))
    return out


def pick(items, metric, year, block=None):
    matches = [v for v in items if v.label == metric and v.period == year and (block is None or v.block==block)]
    if len({(v.value,v.unit) for v in matches}) != 1:
        return None
    return matches[0]


def percent(after, before):
    return (100*(after/before-1)).quantize(Decimal('.01')) if before else None


def eps_bridge(question, corp):
    q = compact(question)
    if not ('기본주당' in q and '가중평균' in q):
        return None
    years = sorted(set(int(y) for y in re.findall(r'20\d{2}',question)))
    if len(years)!=2:
        return None
    meta = annual(corp,years[-1])
    if not meta:
        return None
    sources = sections(corp,meta,lambda p: '주당' in p and p.endswith('(연결)'))
    for source in sources:
        obs = observations(source['text'],years[-1])
        fields = ['보통주당기순이익(지배기업소유주지분)','가중평균유통보통주식수(주)','기본주당이익']
        cells = [[pick(obs,f,y) for f in fields] for y in years]
        if any(v is None for row in cells for v in row):
            continue
        if any([v.unit for v in row] != ['원','주','원'] for row in cells):
            continue
        old,new = [[v.value for v in row] for row in cells]
        if any(abs(income/shares-eps)>Decimal('.5') for income,shares,eps in (old,new) if shares):
            continue
        if old[1]<=0 or new[1]<=0 or old[0]<=0 or old[2]<=0:
            continue
        merger = [r for r in sections(corp,meta,lambda p:p.endswith('(연결)'))
                  if '합병기일' in r['text']]
        event_lines = list(dict.fromkeys(line for r in merger for line in r['text'].splitlines()
                                        if '합병기일' in line and str(years[0]) in line))
        stock_lines = list(dict.fromkeys(f'{v.period}년 [{v.block}] {v.line}' for v in obs if v.label in ('합병으로인한신주발행','기초발행주식')))
        answer = [f"{corp} {years[-1]}년 사업보고서의 연결 주당순이익 주석에 비교표시된 수치를 사용합니다."]
        for y,(income,shares,eps) in zip(years,(old,new)):
            answer.append(f"{y}년 지배주주 보통주 순이익 {fmt(income)}원, 가중평균유통보통주식수 {fmt(shares)}주, 기본 EPS {fmt(eps)}원입니다. 순이익 / 가중평균주식수는 공시 EPS와 원 단위 반올림 범위에서 일치합니다.")
        pi,pe,ps = percent(new[0],old[0]),percent(new[2],old[2]),percent(new[1],old[1])
        answer.append(f"순이익 증감률 {pi}%, 기본 EPS 증감률 {pe}%, 가중평균주식수 증감률 {ps}%입니다.")
        answer.append("기본 EPS는 순이익을 가중평균주식수로 나눈 값이므로, 이익의 변화와 분모의 변화를 함께 반영합니다. 기말 발행주식수 자체를 EPS 분모로 쓰지 않습니다.")
        if event_lines and stock_lines:
            answer.append('합병 시점에 관한 공시: '+' '.join(event_lines))
            answer.append('주식수 변동표의 관련 행:\n'+'\n'.join(stock_lines))
            answer.append('합병 신주는 전기에는 합병 이후 기간만 가중평균에 반영되고 당기에는 기초 주식수에 포함됩니다. 자기주식과 주식선택권 등 다른 변동도 있어 증가분 전체를 합병 하나로 돌리지 않습니다.')
        else:
            answer.append('합병의 개별 기여는 여기서 확정하지 않았습니다.')
        extra = [dict(r,text='\n'.join(event_lines)) for r in merger[:1]] if event_lines else []
        answer.append('근거: '+cite(source))
        return ToolResult(f'{corp} EPS 분모 변화','\n'.join(answer),[source]+extra,
                          f'동일 보고서의 기본 EPS 분자·분모를 검산했다. 순이익 {pi}%, EPS {pe}%, 가중평균주식수 {ps}%',
                          '지정 연도 보고서의 비교표시 기준이며 다른 판본의 EPS나 희석 EPS를 섞지 않았다')
    return None


def income_statement(corp,meta,connected):
    return sections(corp,meta,lambda p: bool(re.search(r'>2-2\..*손익계산서$',p)) if connected
                    else bool(re.search(r'>4-2\..*손익계산서$',p)))


def basis_bridge(question,corp):
    if not all(k in compact(question) for k in ('별도','연결','당기순이익')):
        return None
    years = set(int(y) for y in re.findall(r'20\d{2}',question))
    if len(years)!=1:
        return None
    y=years.pop(); meta=annual(corp,y)
    if not meta:
        return None
    sources=[income_statement(corp,meta,b) for b in (True,False)]
    if any(len(s)!=1 for s in sources):
        return None
    sources=[s[0] for s in sources]; obs=[observations(s['text'],y) for s in sources]
    fields=['당기순이익(손실)','영업이익','기타이익','기타손실','금융수익','금융비용','법인세비용(수익)']
    data=[[pick(o,f,y) for f in fields] for o in obs]
    if any(v is None for row in data for v in row) or len({v.unit for row in data for v in row})!=1:
        return None
    con,sep=[[v.value for v in row] for row in data]; unit=data[0][0].unit
    contributions=[(f,(s-c)*sign) for f,c,s,sign in zip(fields[1:],con[1:],sep[1:],[1,1,-1,1,-1,-1])]
    # 연결에만 존재하는 지분법 항목도 차이의 구성에 포함한다.
    equity=pick(obs[0],'지분법이익',y)
    if equity and not pick(obs[1],'지분법이익',y):
        contributions.append(('연결에만 표시된 지분법이익의 차이 기여',-equity.value))
    residual=sep[0]-con[0]-sum(v for _,v in contributions)
    if residual!=0:
        return None
    biggest=max(contributions,key=lambda v:abs(v[1]))
    notes=sections(corp,meta,lambda p:'기타수익및기타비용' in p and not p.endswith('(연결)'))
    dividend=None
    for s in notes:
        dividend=pick(observations(s['text'],y),'배당금수익',y)
        if dividend:
            sources.append(s);break
    answer=[f'{y}년 연결 당기순이익 {fmt(con[0])}{unit}, 별도 {fmt(sep[0])}{unit}, 별도 − 연결 = {fmt(sep[0]-con[0])}{unit}입니다.',
            '순이익 차이의 구성(수익은 별도 − 연결, 비용은 그 차이의 반대 부호):']
    answer += [f'{f}: {fmt(v)}{unit}' for f,v in contributions]
    answer.append(f'구성항목 합계는 {fmt(sum(v for _,v in contributions))}{unit}이며 순이익 차이와 일치합니다. 절대 기여가 가장 큰 항목은 {biggest[0]} {fmt(biggest[1])}{unit}입니다.')
    if dividend:
        answer.append(f'별도 기타이익 {fmt(sep[2])}{unit} 중 배당금수익은 {fmt(dividend.value)}{unit}이며 비중은 {dividend.value/sep[2]*100:.2f}%입니다. 연결 기타이익은 {fmt(con[2])}{unit}입니다.')
    policy=sections(corp,meta,lambda p:p.endswith('(연결)'))
    policy_sources=[dict(s,text='\n'.join(line for line in s['text'].splitlines()
                                        if '내부' in line and '제거' in line and ('수익' in line or '거래' in line))) for s in policy]
    policy_sources=[s for s in policy_sources if s['text']]
    lines=[line for s in policy_sources for line in s['text'].splitlines()]
    if lines:
        sources.extend(policy_sources)
        answer.append('연결 작성 원칙의 원문: '+lines[0])
    answer.append('두 손익계산서의 배당·기타이익 차이는 연결 범위와 내부거래 제거를 함께 봐야 합니다. 기타이익 차이 전액을 특정 종속기업 배당 제거액으로 확정하지 않았으며, 순이익 차이와도 동일하지 않습니다.')
    answer.append('근거: '+cite(sources[0]))
    return ToolResult(f'{corp} 별도·연결 손익 차이','\n'.join(answer),sources,
                      f'동일 기간·단위의 손익 구성 차이를 합산한 결과 순이익 차이 {fmt(sep[0]-con[0])}{unit}와 일치했다',
                      '손익계산서 간 수치 대사이며 개별 종속기업 배당의 제거분 전액을 식별한 것은 아니다')


def financial_revenue(question,corp):
    if not all(k in compact(question) for k in ('최상단','수익항목','제조업')):
        return None
    years=set(int(y) for y in re.findall(r'20\d{2}',question))
    if len(years)>1:
        return None
    meta=annual(corp,next(iter(years),None))
    if not meta:
        return None
    y=int(re.search(r'\((20\d{2})',meta['report_nm'])[1])
    sources=income_statement(corp,meta,True)
    if len(sources)!=1:
        return None
    obs=observations(sources[0]['text'],y)
    current=[v for v in obs if v.period==y]
    if not current or current[0].label!='영업수익':
        return None
    fields=['영업수익','수수료수익','이자수익','당기손익-공정가치측정금융상품관련이익','당기손익-공정가치측정금융상품관련손실']
    vals=[pick(obs,f,y) for f in fields]
    if any(v is None for v in vals):
        return None
    answer=[f'{corp} {y}년 연결 손익계산서의 최상단은 영업수익 {fmt(vals[0].value)}{vals[0].unit}입니다.']
    answer += [f'{v.label}: {fmt(v.value)}{v.unit}' for v in vals[1:]]
    answer.append('수수료와 이자는 서로 다른 금융활동의 수익이며 금융상품 관련 이익·손실도 별도 행에 표시됩니다. 이자수익의 하위 행을 이자수익에 다시 더하면 중복됩니다. 금융상품 이익과 손실을 임의로 상계한 순액을 위 영업수익과 같은 값으로 볼 수도 없습니다. 제조업 제품 판매액과 구성 및 총액·순액 표시를 확인하지 않고 기업 규모를 단순 비교하면 안 됩니다.')
    answer.append('근거: '+cite(sources[0]))
    return ToolResult(f'{corp} 영업수익 계층','\n'.join(answer),sources,
                      f'연결 손익계산서 첫 수익 행은 영업수익이고 수수료·이자·금융상품 관련 행이 함께 기재돼 있었다',
                      f'{y}년 연결 공시 양식의 구성 비교이며 제조기업과 경제적 규모를 동일 기준으로 환산한 것은 아니다')


def run(question,corp):
    q=compact(question)
    routes=((('기본주당','가중평균'),eps_bridge),
            (('별도','연결','당기순이익'),basis_bridge),
            (('최상단','수익항목','제조업'),financial_revenue),
            (('매출','계속영업','중단영업'),discontinued),
            (('계약자산','계약부채'),contract_positions))
    for terms,tool in routes:
        if not all(term in q for term in terms):
            continue
        result=tool(question,corp)
        if result:
            result.basis = {
                eps_bridge: '기본 EPS의 분모는 기말 발행주식수가 아니므로 동일 보고서의 기본 EPS·귀속이익·가중평균주식수와 비교표시 기간을 맞췄다',
                basis_bridge: '별도와 연결은 회계 범위가 다르므로 한 항목의 차이를 순이익 차이로 대입하지 않고 수익·비용의 부호를 적용해 구성 전체를 대사했다',
                financial_revenue: '하위 수익항목을 최상단 총액으로 오독하지 않도록 연결 손익계산서의 행 순서와 수익·비용 구분을 확인했다',
                discontinued: '분할 후 비교표시와 원보고는 범위가 다를 수 있어 재작성 주석과 계속·중단영업 구분을 확인했다',
                contract_positions: '회사 전체 계정과 계약별 포지션을 혼동하지 않도록 동일 주석의 기간별 합계 열만 비교했다',
            }[tool]
            return result
        # 지원하는 비교에서 좌표를 확보하지 못하면 생성 단계로 우회하지 않는다.
        return ToolResult(f'{corp} 비교 기준 미확정',
                          '요구한 비교를 위한 동일 보고서·기간·회계 기준·단위의 표 값을 확정하지 못했습니다. 이 상태에서 숫자를 합치거나 다른 기준의 값으로 대체하지 않았습니다. 대상 보고서의 날짜 또는 접수번호를 지정하면 확인 범위를 좁힐 수 있습니다.',
                          [],'비교표의 필수 좌표 또는 검산 조건을 충족하지 못해 수치 생성을 중단했다',
                          '파싱 및 검산 실패이며 공시 원문에 해당 정보가 없다고 판정한 것은 아니다')
    return None


def contract_positions(question,corp):
    if not all(k in compact(question) for k in ('계약자산','계약부채')):
        return None
    asked=set(int(y) for y in re.findall(r'20\d{2}',question))
    samples=[]; checked=[]; skipped=[]
    for meta in AGG.docs_for(corp,{'report':'사업보고서'}):
        m=re.search(r'\((20\d{2})\.12\)',meta['report_nm'])
        if not m or (asked and int(m[1]) not in asked):
            continue
        y=int(m[1])
        if meta['rcept_no']!=annual(corp,y)['rcept_no']:
            continue
        for source in sections(corp,meta,lambda p:'건설계약' in p and p.endswith('(연결)')):
            checked.append(source); period=None; col=None; unit=None; found={}; active=False
            for line in source['text'].splitlines():
                s=compact(line)
                if '계약자산및계약부채' in s:
                    active=True;period=None;col=None
                if '(3)' in s:
                    active=False
                if not active or not line.strip().startswith('|'):
                    continue
                cells=[c.strip() for c in line.strip().strip('|').split('|')]
                if cells[0] in ('당기','전기'):
                    period=y-(cells[0]=='전기');col=None
                u=re.search(r'단위[:：](백만원|천원|원)',s)
                if u:
                    unit=u[1]
                totals=[i for i,c in enumerate(cells) if compact(c)=='제품과용역합계']
                if len(totals)==1:
                    col=totals[0]
                if compact(cells[0]) in ('유동계약자산','유동계약부채') and col is not None and col<len(cells) and period and unit:
                    v=number(cells[col])
                    if v is not None:
                        found.setdefault((period,compact(cells[0])),set()).add((v,unit))
            keys=[(p,f) for p in (y-1,y) for f in ('유동계약자산','유동계약부채')]
            if any(len(found.get(k,set()))!=1 for k in keys):
                skipped.append(meta['rcept_no']);continue
            data=[next(iter(found[k])) for k in keys]
            if len({u for v,u in data})!=1:
                continue
            samples.append((y,[v for v,u in data],data[0][1],source))
    if not checked:
        return None
    answer=[]
    for y,(a0,l0,a1,l1),unit,s in samples:
        answer.append(f'{y-1}→{y}년 연결 건설계약 주석의 제품과 용역 합계: 유동계약자산 {fmt(a0)}→{fmt(a1)}{unit} (증감 {fmt(a1-a0)}), 유동계약부채 {fmt(l0)}→{fmt(l1)}{unit} (증감 {fmt(l1-l0)}). 접수번호 {s["rcept_no"]}')
    both=[y for y,(a0,l0,a1,l1),_,_ in samples if a1>a0 and l1>l0]
    answer.append('위 비교에서 두 행이 함께 증가한 연도: '+(', '.join(map(str,both)) if both else '확인되지 않음')+'.')
    answer.append('이는 읽힌 두 행의 비교입니다. 선수금 등 다른 계약부채 항목과 비교 범위를 합치거나, 미추출 연도까지 동시에 증가한 적이 없다고 일반화하지 않았습니다.')
    if skipped:
        answer.append('동일 합계 열을 확보하지 못한 보고서: '+', '.join(skipped))
    answer.append('계약자산과 계약부채는 이행과 대금 지급의 관계를 계약별로 표시하는 항목입니다. 서로 다른 계약의 자산과 부채를 회사 전체에서 임의로 상계하는 개념은 아니며, 서로 다른 계약에서 양쪽 잔액이 동시에 늘어날 수 있습니다. 이 원칙만으로 특정 연도의 동시 증가가 입증되지는 않습니다.')
    policies=[]
    for s in sections(corp,annual(corp),lambda p:p.endswith('(연결)')):
        lines=[l for l in s['text'].splitlines() if ('계약자산' in l or '계약부채' in l) and ('지급' in l or '이행' in l) and len(l)<5000]
        if lines:
            policies.append(dict(s,text='\n'.join(lines)))
    if policies:
        answer.append('관련 회계정책 출처: '+cite(policies[0]))
    return ToolResult(f'{corp} 계약별 표시 범위','\n'.join(answer),checked+policies,
                      f'연결 건설계약 주석 {len(checked)}개 중 같은 합계 열로 비교한 연도쌍은 {len(samples)}개, 동시 증가 확인은 {len(both)}개였다',
                      '미추출 연도와 다른 계약부채 항목을 포함한 전체 회사의 동시 증가 여부는 확정하지 않았다')


def discontinued(question,corp):
    q=compact(question)
    if not all(k in q for k in ('매출','계속영업','중단영업')):
        return None
    years=set(int(y) for y in re.findall(r'20\d{2}',question))
    # 분할 당기라고 명시한 주석을 가진 보고서를 찾는다. 이후 보고서의 재서술과 구분한다.
    for meta in AGG.docs_for(corp,{'report':'사업보고서'}):
        m=re.search(r'\((20\d{2})\.12\)',meta['report_nm'])
        if not m or (years and int(m[1]) not in years):
            continue
        y=int(m[1]); latest=annual(corp,y)
        if meta['rcept_no']!=latest['rcept_no']:
            continue
        notes=sections(corp,meta,lambda p:'중단영업' in p and p.endswith('(연결)'))
        notes=[s for s in notes if '재작성' in s['text'] and '당기 분할신설회사' in s['text']]
        main=income_statement(corp,meta,True)
        if len(notes)!=1 or len(main)!=1:
            continue
        obs=observations(main[0]['text'],y)
        vals=[pick(obs,f,y) for f in ('계속영업순이익','중단영업손익','당기순이익(손실)')]
        current,comparison=pick(obs,'매출액',y),pick(obs,'매출액',y-1)
        if any(v is None for v in vals+[current,comparison]) or len({v.unit for v in vals+[current,comparison]})!=1:
            continue
        if vals[0].value+vals[1].value!=vals[2].value:
            continue
        oldmeta=annual(corp,y-1); oldsrc=income_statement(corp,oldmeta,True) if oldmeta else []
        original=pick(observations(oldsrc[0]['text'],y-1),'매출액',y-1) if len(oldsrc)==1 else None
        business=[]; business_value=None
        if original is None:
            business=sections(corp,meta,lambda p:'II.사업의내용>' in p and '매출및수주상황' in p,financial_only=False)
            if len(business)==1:
                business_value=pick(observations(business[0]['text'],y),'연결후매출액합계',y-1)
        answer=[f'{y}년 보고서의 연결 손익계산서와 인적분할·중단영업 주석을 기준으로 비교합니다.',
                f'{y}년 계속영업 매출액 {fmt(current.value)}{current.unit}, 같은 보고서에 비교표시된 {y-1}년 매출액 {fmt(comparison.value)}{comparison.unit}입니다.']
        if original and original.unit==comparison.unit:
            answer.append(f'{y-1}년 보고서의 원보고 매출액은 {fmt(original.value)}{original.unit}입니다. 이후 보고서의 재작성 비교매출과 {fmt(original.value-comparison.value)}{original.unit} 차이가 있어 같은 범위의 값으로 대입하면 안 됩니다. 원보고 출처: '+cite(oldsrc[0]))
        elif business_value:
            answer.append(f'같은 보고서 사업의 내용에는 {y-1}년 연간 연결 후 매출액 합계가 {fmt(business_value.value)}{business_value.unit}으로 기재돼 있습니다. 이를 재무제표의 재작성 비교매출과 구분합니다. 과거 사업보고서 원본을 직접 확보한 것은 아닙니다. 사업의 내용 출처: '+cite(business[0]))
        else:
            answer.append('원보고 매출액과의 수치 대사는 확인하지 못했습니다.')
        explanation=next(line for line in notes[0]['text'].splitlines() if '재작성' in line)
        answer.append('재작성 근거: '+explanation)
        answer.append(f'순이익 검산: 계속영업 {fmt(vals[0].value)} + 중단영업 {fmt(vals[1].value)} = 전체 {fmt(vals[2].value)}{vals[0].unit}. 중단영업 손익을 삭제한 것이 아니라 별도 표시한 것입니다. 단위 환산 전에 원문 정밀도로 합산했습니다.')
        answer.append('근거: '+cite(main[0]))
        return ToolResult(f'{corp} 분할 전후 비교범위','\n'.join(answer),main+notes+oldsrc+business,
                          f'재작성 주석을 확인하고 계속영업·중단영업 합계 {fmt(vals[2].value)}{vals[0].unit}를 검산했다'+(' ; 원보고 매출은 미확인이다' if original is None else ''),
                          '분할 후 보고서의 비교표시 기준이며 수치 차이 전체를 성장이나 감소로 해석하지 않았다')
    return None
