import unittest
from decimal import Decimal
import struct_ops as S
import http.client
import json
import threading
from unittest.mock import patch
import server


class ArithmeticRenderingTests(unittest.TestCase):
    def test_nonzero_residual_cannot_be_complete_offset(self):
        for value in [Decimal('-17400'), Decimal('17400')]:
            ds = [S.Derivation('diff', '차감(합계 금액): 검산', value, '원', ['a','b'])]
            answer = S.render_arithmetic(ds)
            self.assertIn('완전히 상쇄된 것은 아닙니다', answer)
            self.assertIn('17,400원', answer)
            self.assertIn('감소' if value < 0 else '증가', answer)

    def test_zero_total_even_with_nonzero_components(self):
        ds = [S.Derivation('diff', '차감(금액, A)', Decimal('5'), '원', []),
              S.Derivation('diff', '차감(금액, B)', Decimal('-5'), '원', []),
              S.Derivation('diff', '차감(합계 금액)', Decimal('0'), '원', [])]
        self.assertIn('총액에서 완전히 상쇄됩니다', S.render_arithmetic(ds))

    def test_ratios_retain_existing_path(self):
        self.assertIsNone(S.render_arithmetic([S.Derivation('ratio','비율',Decimal('57.01'),'%',[])]))
        self.assertIsNone(S.render_arithmetic([]))


class HttpResponseTests(unittest.TestCase):
    def test_complete_response_and_cached_retry(self):
        response = dict(question_id='retry-test', question='test', retrieved_context='', think_trace='', answer='ok')
        server._cache.clear()
        srv = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(server, 'safe_answer', return_value=response.copy()) as answer:
                bodies=[]
                for _ in range(3):
                    conn=http.client.HTTPConnection('127.0.0.1', srv.server_port, timeout=5)
                    conn.request('GET', '/answer?question_id=retry-test&question=test')
                    resp=conn.getresponse()
                    body=resp.read()
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(int(resp.getheader('Content-Length')), len(body))
                    self.assertEqual(resp.getheader('Connection'), 'close')
                    self.assertEqual(json.loads(body), response)
                    bodies.append(body)
                    conn.close()
                self.assertEqual(answer.call_count, 1)
                self.assertEqual(len(set(bodies)), 1)
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join()
            server._cache.clear()


if __name__ == '__main__':
    unittest.main()
