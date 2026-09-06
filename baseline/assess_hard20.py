"""검토 대상 10문항의 필수 항목 누락을 찾는다. 의미 검토는 별도로 기록한다."""
import argparse
import json
import re
from pathlib import Path


REQUIRED = {
    'H01': ['2,682,737,598,000','317,262,452,400','2,696,357,832,000','303,642,201,000',
            '3,000,000,050,400','3,000,000,033,000','17,400','20241118000328','20250218001596'],
    'H02': ['13,620,234,000','13,620,251,400','17,400','보통주','기타주식','20241118000328','20250218001596'],
    'H03': ['1−2−3−4−5+6','차감','취득원가','복원','처분대금이나 처분이익이 아닙니다'],
    'H04': ['min(max(A, B), C)','5,014,462','6,525,123','59,697,825','691,203','406,507','8,228,867','20241118000328'],
    'H09': ['24,858,075','43,601,051','57.01','57.0','반올림','일치','백만원','억원'],
    'H10': ['39.01','57.01','19.10','13.07','영업이익률','보다 높다','20260310002820'],
    'H11': ['문서 수로는 1건','계약금액 변경 1회','계약기간 변경 1회','총 2회','변경된 표 행'],
    'H13': ['2024-07-24','2028-06-24','2028-07-24','2026-09-01','포함됩니다','미상환','날짜 조건'],
    'H14': ['440,000,000,000','275,000','206,250','75%','1,600,000','2,133,333','533,333','68,750','단주','가정'],
    'H19': ['7,595','2,795','392,269','17,509','7,121','421,699','차감 항목','중복','20260311004517'],
}


def assess(rows):
    checks=[]
    for row in rows:
        qid=row.get('qid') or row['question_id']
        if qid not in REQUIRED:
            checks.append(dict(qid=qid,run=row.get('run'),status='not_reviewed'))
            continue
        answer=row['answer']
        missing=[s for s in REQUIRED[qid] if s not in answer]
        answer_refs=set(re.findall(r'(?<!\d)\d{14}(?!\d)',answer))
        context_refs=set(re.findall(r'(?<!\d)\d{14}(?!\d)',row['retrieved_context']))
        source_ok=bool(answer_refs) and answer_refs <= context_refs
        contradiction='수치를 확정하지 못했습니다' in answer or '확인할 수 없습니다' in answer
        api_ok=all(isinstance(row.get(k),str) for k in ('question_id','question','retrieved_context','think_trace','answer'))
        checks.append(dict(qid=qid,run=row.get('run'),missing=missing,source_ok=source_ok,
                           contradiction=contradiction,api_ok=api_ok,
                           status='review_candidate' if not missing and source_ok and not contradiction and api_ok else 'check_failed'))
    ready=[qid for qid in REQUIRED if len([r for r in checks if r['qid']==qid])==3
           and all(r['status']=='review_candidate' for r in checks if r['qid']==qid)]
    return dict(notice='필수 항목 및 출처 검사는 문장 의미의 정답 판정을 대신하지 않는다. FOLLOWUP_EVALUATION.md의 원문 대조 기록과 함께 사용한다.',
                three_run_candidates=ready,three_run_candidate_count=len(ready),rows=checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    args=parser.parse_args()
    result=assess(json.loads(Path(args.source).read_text(encoding='utf-8')))
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result['three_run_candidate_count'],result['three_run_candidates'])
