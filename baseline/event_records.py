"""공시의 사건 필드와 명시된 정정 대상 날짜로 판본을 연결한다."""
from collections import defaultdict
from decimal import Decimal
import re

import aggregate_tools as AGG
from disclosure_tools import ToolResult, compact, document, rows, date_values, unique_field, cite, fmt


def field_text(text, field):
    result=[]
    for cells,line in rows(text):
        if field in compact(cells[0]):
            # 값이 두 번 반복되는 서식과 정정 전후의 별도 값을 모두 보존한다.
            result.extend(c for c in cells[1:] if c and field not in compact(c) and c!='-')
    return list(dict.fromkeys(result))


def contract_graph(sources):
    nodes={s['rcept_no']:s for s in sources}
    descriptors={}
    for no,s in nodes.items():
        names=field_text(s['text'],'체결계약명')
        counterparties=field_text(s['text'],'3.계약상대')
        target,_=unique_field(s['text'],['정정관련공시서류제출일'],dates=True)
        descriptors[no]=(set(map(compact,names)),set(map(compact,counterparties)),target)
    parents={}; unresolved=[]
    for no,s in nodes.items():
        if '정정' not in s['report_nm']:
            continue
        names,parties,target=descriptors[no]
        matches=[]
        if target:
            for other,t in nodes.items():
                onames,oparties,_=descriptors[other]
                if other<no and t['rcept_dt']==target.strftime('%Y%m%d') and names&onames and parties&oparties:
                    matches.append(other)
        if len(matches)==1:
            parents[no]=matches[0]
        else:
            unresolved.append(no)
    groups=defaultdict(list)
    for no in nodes:
        root=no
        while root in parents:
            root=parents[root]
        groups[root].append(no)
    return nodes,descriptors,parents,groups,unresolved


def correction_history(question,corp):
    q=compact(question)
    if not all(k in q for k in ('계약','정정','가장','횟수')):
        return None
    metas=AGG.docs_for(corp,{'report':'공급계약'})
    sources=[document(corp,m) for m in metas]
    nodes,desc,parents,groups,unresolved=contract_graph(sources)
    complete={root:ids for root,ids in groups.items() if '정정' not in nodes[root]['report_nm']}
    if not complete:
        return ToolResult(f'{corp} 정정 연결 한계','원공시까지 명시적으로 연결된 계약을 확인하지 못해 최다 정정 계약을 확정하지 않았습니다.',sources,
                          f'원장 {len(sources)}건을 확인했으나 완결 체인이 없었다','계약명만 같은 문서를 동일 사건으로 묶지 않았다')
    maximum=max(len(ids)-1 for ids in complete.values())
    winners=[(root,ids) for root,ids in complete.items() if len(ids)-1==maximum]
    answer=[f'코퍼스의 단일판매·공급계약 공시 {len(nodes)}건을 접수번호로 중복 제거하고, 정정 대상 제출일·계약명·계약상대가 일치하는 문서를 연결했습니다.',
            f'원공시까지 연결된 {len(complete)}개 계약 중 관측된 최대 정정 횟수는 {maximum}회이며 동률은 {len(winners)}개입니다.']
    picked=[]
    for root,ids in winners:
        name=' / '.join(field_text(nodes[root]['text'],'체결계약명'))
        answer.append('계약: '+name)
        for i,no in enumerate(sorted(ids,key=lambda n:(nodes[n]['rcept_dt'],n))):
            answer.append(f'{"원공시" if i==0 else "관측 정정 "+str(i)+"차"}: {nodes[no]["rcept_dt"]}, 접수번호 {no}')
            picked.append(nodes[no])
    partial={root:ids for root,ids in groups.items() if root not in complete}
    if partial:
        observed=max(len(ids) for ids in partial.values())
        answer.append(f'원공시가 미연결인 계약 묶음도 포함하면 정정 문서가 가장 많이 관측된 묶음은 {observed}건입니다. 위의 완결 체인 순위와 다릅니다.')
        for root,ids in partial.items():
            if len(ids)!=observed:
                continue
            answer.append('미완결 계약: '+' / '.join(field_text(nodes[root]['text'],'체결계약명')))
            answer.append('관측 정정일 및 접수번호: '+', '.join(nodes[no]['rcept_dt']+' ('+no+')' for no in sorted(ids)))
            picked.extend(nodes[no] for no in ids)
    limit=f'원공시까지 연결하지 못한 정정 문서 {len(unresolved)}건이 있으므로 전체 역사에서의 최다 계약이나 실제 정정 차수는 확정하지 않았다' if unresolved else '배포 코퍼스에 수록된 판본 기준이며 수집기간 밖 정정은 포함하지 않았다'
    answer.append(limit+'.')
    if unresolved:
        answer.append('미연결 접수번호: '+', '.join(unresolved))
        # 근거에 미연결 문서의 대상 날짜와 식별 필드를 제공한다.
        picked += [dict(nodes[no],text='\n'.join(line for line in nodes[no]['text'].splitlines() if any(k in compact(line) for k in ('체결계약명','3.계약상대','정정관련공시서류제출일')))) for no in unresolved]
    return ToolResult(f'{corp} 계약 판본 연결','\n'.join(answer),picked,
                      f'원장 {len(nodes)}건 중 명시적 정정 연결 {len(parents)}개, 원공시 포함 계약 {len(complete)}개, 완결 체인 내 최대 {maximum}회'+(f', 미완결 묶음의 최대 관측 정정 문서는 {observed}건' if partial else ''),limit)


def disposal(question,corp):
    q=compact(question)
    if not all(k in q for k in ('처분','소각','주식')):
        return None
    years=set(re.findall(r'20\d{2}',question))
    metas=AGG.docs_for(corp,{'report':'자기주식처분결정'})
    if years:
        metas=[m for m in metas if m['rcept_dt'][:4] in years]
    events=[]
    for meta in metas:
        s=document(corp,meta)
        common,_=unique_field(s['text'],['1.처분예정주식(주)','보통주식'])
        other,_=unique_field(s['text'],['1.처분예정주식(주)','기타주식'])
        purpose=field_text(s['text'],'5.처분목적')
        if common is not None and other is not None and len(purpose)==1:
            events.append((s,common,other,purpose[0]))
    if not events:
        return None
    answer=['날짜가 지정되지 않은 경우 수록된 처분결정들을 공시별로 제시합니다. 정정 전후가 같은 사건일 수 있으므로 아래 수량을 전부 더하지 않습니다.']
    for s,c,o,p in events:
        answer.append(f'{s["rcept_dt"]} 보통주 {fmt(c)}주, 기타주식 {fmt(o)}주. 목적: {p}. 접수번호 {s["rcept_no"]}')
    answer.append('이는 자기주식 처분결정에 기재된 예정 수량이며 실제 지급 완료 수량으로 확정하지 않았습니다. 기존 자기주식을 임직원 등에게 이전하는 처분 자체는 주식 소멸이 아니므로, 자기주식 보유량 감소와 발행주식총수 감소를 구분해야 합니다. 소각은 주식 자체를 없애는 별도 사건입니다. 취득·처분·소각의 수량이나 목적을 서로 대입하지 않습니다.')
    return ToolResult(f'{corp} 처분과 소각 구분','\n'.join(answer),[e[0] for e in events],
                      f'처분결정 {len(metas)}건 중 수량과 목적을 읽은 {len(events)}건을 개별 공시로 제시했다',
                      '처분 예정과 실행을 구분하며 특정 소각결정과 동일 사건이라고 합치지 않았다')


def run(question,corp):
    for tool in (correction_history,disposal,purpose_allocation):
        result=tool(question,corp)
        if result:
            result.basis = {correction_history:'정정 문서 수와 최초 공시 이후 차수는 다르므로 원공시 연결 여부를 나누어 집계했다',
                            disposal:'처분 목적과 수량은 같은 처분결정에서 읽고 주식 소멸 여부가 다른 소각과 분리했다',
                            purpose_allocation:'목적별 금액을 연간 취득액으로 확대하지 않고 해당 결정의 총액과 부분합 차이를 검산했다'}[tool]
            return result
    return None


def purpose_values(text):
    pattern=r'(?:(?P<jo>[\d,]+)조\s*)?(?P<eok>[\d,]+)억원(?:은|는)\s*(?P<label>[^.,\n]{2,80}?)목적'
    out=[]
    for m in re.finditer(pattern,text):
        value=(Decimal((m['jo'] or '0').replace(',',''))*10000+Decimal(m['eok'].replace(',','')))*100000000
        name=re.sub(r'(?:을|를|의)\s*$','',m['label']).strip()
        if (name,value) not in out:
            out.append((name,value))
    return out


def purpose_allocation(question,corp):
    q=compact(question)
    if not all(k in q for k in ('취득','목적','비중')):
        return None
    years=set(re.findall(r'20\d{2}',question))
    for meta in reversed(AGG.docs_for(corp,{'report':'자기주식취득결정'})):
        if years and meta['rcept_dt'][:4] not in years:
            continue
        source=document(corp,meta)
        purposes=purpose_values(source['text'])
        if len(purposes)<2:
            continue
        c,_=unique_field(source['text'],['2.취득예정금액(원)','보통주식'])
        o,_=unique_field(source['text'],['2.취득예정금액(원)','기타주식'])
        if c is None or o is None:
            continue
        total=c+o; parts=sum(v for _,v in purposes)
        if total<=0 or parts<=0 or abs(total-parts)/total>Decimal('.001'):
            continue
        answer=[f'{source["rcept_dt"]} 자기주식 취득결정 한 건의 목적별 배분입니다. 연간 전체 취득액 집계가 아닙니다.']
        answer.extend(f'{name}: {fmt(v)}원, 목적별 기재금액 합계 대비 {v/parts*100:.2f}%' for name,v in purposes)
        answer.append(f'부분합 {fmt(parts)}원, 취득예정금액 표의 보통주 {fmt(c)} + 기타주식 {fmt(o)} = {fmt(total)}원. 차이는 {fmt(total-parts)}원이며 총액 대비 {abs(total-parts)/total*100:.4f}%입니다. 목적별 서술 금액의 표시 정밀도가 낮아 부분합과 표의 총액이 정확히 같지는 않습니다. 위 비중의 분모는 목적별 기재금액의 합계입니다.')
        answer.append('취득 예정의 목적 배분이며 실제 취득 완료나 임직원 지급 완료를 의미하지 않습니다.\n근거: '+cite(source))
        return ToolResult(f'{corp} 목적별 취득 배분','\n'.join(answer),[source],
                          f'한 결정의 목적별 부분합 {fmt(parts)}원과 표 총액 {fmt(total)}원을 대사했다',
                          f'목적별 표시금액과 표 총액 간 차이 {fmt(total-parts)}원을 남겼으며 연간 합계로 확대하지 않았다')
    return None
