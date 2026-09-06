"""공개 API를 순차 호출한다. 반복마다 새 ID를 사용해 캐시 적중을 피한다."""
import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('endpoint')
    parser.add_argument('output')
    parser.add_argument('--ids', default='H14,H01,H02,H03,H04,H05,H06')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('기존 결과를 덮어쓰지 않습니다')
    questions = json.loads(Path(__file__).with_name('hard20_questions.json').read_text(encoding='utf-8'))
    questions = {q['qid']: q['question'] for q in questions}
    batch = uuid.uuid4().hex[:12]
    results = []
    for qid in args.ids.split(','):
        for run in range(1, (1 if qid == 'H14' else 3) + 1):
            request_id = f'PUBLIC-{batch}-{qid}-{run}'
            url = args.endpoint + '?' + urlencode({'question_id': request_id, 'question': questions[qid]})
            start = time.monotonic()
            proc = subprocess.run(['curl.exe', '--max-time', '300', '-sS', '-w', '\n%{http_code}', url], capture_output=True)
            row = dict(qid=qid, run=run, request_id=request_id, sec=round(time.monotonic()-start, 2))
            try:
                body, status = proc.stdout.decode('utf-8').rsplit('\n', 1)
                response = json.loads(body)
                fields = {'question_id', 'question', 'retrieved_context', 'think_trace', 'answer'}
                row.update(response)
                row['http_status'] = int(status)
                row['api_ok'] = (proc.returncode == 0 and status == '200' and set(response) == fields
                                 and all(isinstance(v, str) for v in response.values())
                                 and response['question_id'] == request_id and response['question'] == questions[qid])
            except (ValueError, TypeError):
                row.update(api_ok=False, error=proc.stderr.decode('utf-8', errors='replace'), raw=proc.stdout.decode('utf-8', errors='replace'))
            results.append(row)
            output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"{qid} run={run} sec={row['sec']} api_ok={row['api_ok']}", flush=True)
            if not row['api_ok']:
                raise SystemExit('API 규격 실패: 저장 후 중단')


if __name__ == '__main__':
    main()
