"""개발·테스트용 가짜 OpenAI 호환 LLM 서버 (오케스트레이터/교사 역할 흉내).

python -m tests.mock_llm_server --port 18080
실제 AI 없이 파이프라인 전체(명세→생성→평가→학습→평가)를 검증할 때 사용한다.
"""
import argparse
import itertools
import json
import random
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_counter = itertools.count(1)
_rng = random.Random(7)


def _qa():
    a, b = _rng.randint(2, 99), _rng.randint(2, 99)
    n = next(_counter)
    return f"[{n}] {a} 더하기 {b}는 얼마인가요?", f"{a} 더하기 {b}는 {a + b}입니다."


def respond(messages):
    text = "\n".join(m.get("content", "") for m in messages)
    last = messages[-1]["content"] if messages else ""
    if "Finalize the training plan" in last:
        return json.dumps({"title": "Addition teacher", "goal": "Answer two-digit addition correctly",
                           "goal_type": "adaptation", "domain": "Addition", "language": "Korean",
                           "answer_style": "'A 더하기 B는 C입니다.' format, one sentence",
                           "target_samples": 120, "student_model_comment": "A small model is enough",
                           "evaluation_criteria": ["Correct sum", "Follows the format"],
                           "plan_markdown": "# Plan\nTeach two-digit addition in a fixed one-sentence format.",
                           "user_summary_markdown": "두 자리 수 덧셈을 정해진 한 문장 형식으로 답하게 합니다."},
                          ensure_ascii=False)
    if "Translate the values of the JSON below into natural English" in last:
        return json.dumps({"title": "Translated plan", "goal": "Translated goal", "domain": "Translated domain",
                           "answer_style": "One sentence", "student_model_comment": "OK",
                           "evaluation_criteria": ["Correct"], "plan_markdown": "# Translated"})
    if "Write a training-data specification" in last:
        return "```json\n" + json.dumps({"language": "한국어", "system_prompt": "너는 덧셈 선생님이다.",
                                          "answer_format": "'A 더하기 B는 C입니다.'", "avg_answer_tokens": 20,
                                          "categories": [{"name": "쉬운 덧셈", "description": "받아올림 없음", "count": 60,
                                                          "difficulty": {"easy": 1}, "example_question": "12 더하기 3은?"},
                                                         {"name": "받아올림 덧셈", "description": "받아올림 있음", "count": 60,
                                                          "difficulty": {"medium": 0.5, "hard": 0.5},
                                                          "example_question": "58 더하기 67은?"}]},
                                         ensure_ascii=False) + "\n```"
    if "before training and after training" in last:
        return json.dumps({"X": _rng.randint(3, 9), "Y": _rng.randint(3, 9), "reason": "모의 평가"})
    if "You are Teacher 1" in last and "Create exactly" in last:
        n = int(re.search(r"exactly (\d+) items", last).group(1))
        items = []
        for _ in range(n):
            q, a = _qa()
            items.append({"difficulty": "medium", "question": q, "answer": a})
        return "<think>생각 중</think>" + json.dumps({"items": items}, ensure_ascii=False)
    if "model answer" in last and "[Question]" in last:
        q = last.split("[Question]")[1].strip().split("\n")[0]
        m = re.search(r"(\d+) 더하기 (\d+)", q)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            wrong = _rng.random() < 0.1 and "Teacher 2" in last  # 교사2가 가끔 틀림 → 모순/재생성 경로 검증
            return f"{a} 더하기 {b}는 {a + b + (1 if wrong else 0)}입니다."
        return "답변"
    if "quality judge" in last:
        def num(label):
            s = last.split(f"[Answer {label}]")[1].strip().split("\n")[0]
            m = re.search(r"(\d+) 더하기 (\d+)는 (\d+)", s)
            return m and int(m.group(1)) + int(m.group(2)) == int(m.group(3))
        oka, okb = num("A"), num("B")
        sa, sb = (9 if oka else 3), (9 if okb else 3)
        return json.dumps({"question": {"clarity": 9, "relevance": 9, "difficulty_ok": True, "overall": 9},
                           "A": {"accuracy": sa, "completeness": sa, "format": 9, "safety": 10, "overall": sa},
                           "B": {"accuracy": sb, "completeness": sb, "format": 9, "safety": 10, "overall": sb},
                           "contradiction": oka != okb, "winner": "A" if sa >= sb else "B", "reason": "계산 확인"},
                          ensure_ascii=False)
    if "orchestrator" in text:
        return "학생이 어떤 문제를 풀게 하고 싶으신가요?\n<plan_summary>\n- 목표: 미정\n</plan_summary>"
    return "OK"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "mock-orchestrator"}, {"id": "mock-teacher1"}, {"id": "mock-teacher2"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        model = req.get("model", "")
        text = respond(req.get("messages", []))
        if "stall" in model:  # 응답 없이 매달림 (빨간불 확인용)
            time.sleep(600)
            return
        if "loop" in model:  # 같은 말을 반복 (헛돎 확인용)
            text = "같은 말을 반복합니다. " * 200
        if not req.get("stream"):
            self._send({"choices": [{"message": {"role": "assistant", "content": text}}]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        delay = 0.15 if "slow" in model else 0.0
        try:
            for i in range(0, len(text), 4):
                chunk = {"choices": [{"delta": {"content": text[i:i + 4]}}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                if delay:
                    time.sleep(delay)
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()
    print(f"mock LLM on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
