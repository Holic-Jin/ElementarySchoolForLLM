"""PyTorch 학생 모델용 최소 OpenAI 호환 서버 (/v1/models, /v1/chat/completions). 127.0.0.1에서만 대기.

사용: python torch_serve.py --model <dir> [--adapter-path <dir>] --port 8090
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _dtype_kw(dtype):
    """transformers 4.56+는 dtype=, 이전 버전은 torch_dtype= 를 쓴다."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def _chat_ids(tok, msgs, add_generation_prompt):
    """apply_chat_template 결과를 토큰 id 리스트로 통일 (transformers 5는 BatchEncoding 반환)."""
    r = tok.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt, tokenize=True)
    if hasattr(r, "keys"):
        r = r["input_ids"]
    if len(r) and isinstance(r[0], (list, tuple)):
        r = r[0]
    return list(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default="")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    tok_src = args.adapter_path or args.model
    try:
        tok = AutoTokenizer.from_pretrained(tok_src)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **_dtype_kw(dtype))
    if args.adapter_path:
        from pathlib import Path

        if (Path(args.adapter_path) / "adapter_config.json").exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter_path)
        else:  # 전체 파인튜닝 결과
            model = AutoModelForCausalLM.from_pretrained(args.adapter_path, **_dtype_kw(dtype))
    model.to(device).eval()
    lock = threading.Lock()
    name = args.adapter_path or args.model

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"data": [{"id": name, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            msgs = req.get("messages", [])
            max_new = int(req.get("max_tokens") or 512)
            temp = float(req.get("temperature") or 0.0)
            ids = torch.tensor([_chat_ids(tok, msgs, True)], device=device)
            with lock, torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new, do_sample=temp > 0,
                                     temperature=temp if temp > 0 else None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            self._send(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": "stop"}], "model": name})

    print(f"serving on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
