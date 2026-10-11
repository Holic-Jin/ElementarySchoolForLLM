"""LLM 클라이언트 추상화. 주소 + API 키 + 모델명 방식으로 원격/로컬 모두 지원한다.

- openai    : OpenAI 호환 /chat/completions (OpenAI, OpenRouter, Groq, Together, Ollama, LM Studio,
              mlx_lm.server, llama.cpp server, vLLM 등)
- anthropic : Anthropic Messages API
- gemini    : Google Gemini generateContent
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import random
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from ..i18n import T
from .activity import Activity, is_looping
from .cancel import CancelToken
from .jsonutil import extract_json, strip_thinking


@dataclass(frozen=True)
class Preset:
    kind: str
    base_url: str
    tos_url: str = ""  # 이용약관/이용정책 링크 (원격 제공자)
    needs_key: bool = True


# 프리셋 ID는 설정 파일에 저장되는 고정 값, 화면 표시는 PRESET_LABELS(번역)로 한다.
PRESETS: dict[str, Preset] = {
    "ollama": Preset("openai", "http://localhost:11434/v1", needs_key=False),
    "lmstudio": Preset("openai", "http://localhost:1234/v1", needs_key=False),
    "mlx_server": Preset("openai", "http://localhost:8080/v1", needs_key=False),
    "llamacpp": Preset("openai", "http://localhost:8081/v1", needs_key=False),
    "openai": Preset("openai", "https://api.openai.com/v1", "https://openai.com/policies/"),
    "anthropic": Preset("anthropic", "https://api.anthropic.com", "https://www.anthropic.com/legal"),
    "gemini": Preset("gemini", "https://generativelanguage.googleapis.com/v1beta", "https://ai.google.dev/gemini-api/terms"),
    "openrouter": Preset("openai", "https://openrouter.ai/api/v1", "https://openrouter.ai/terms"),
    "groq": Preset("openai", "https://api.groq.com/openai/v1", "https://groq.com/terms-of-use/"),
    "together": Preset("openai", "https://api.together.xyz/v1", "https://www.together.ai/terms-of-service"),
    "deepseek": Preset("openai", "https://api.deepseek.com/v1", "https://platform.deepseek.com/"),
    "custom": Preset("openai", "", needs_key=False),
}
# 이전 버전 설정 파일의 한국어 프리셋 이름 → ID
LEGACY_PRESETS = {"Ollama (로컬)": "ollama", "LM Studio (로컬)": "lmstudio", "mlx_lm.server (로컬)": "mlx_server",
                  "llama.cpp server (로컬)": "llamacpp", "OpenAI": "openai", "Anthropic": "anthropic",
                  "Google Gemini": "gemini", "OpenRouter": "openrouter", "Groq": "groq", "Together AI": "together",
                  "직접 입력": "custom"}


def preset_label(pid: str) -> str:
    return {"ollama": T("Ollama (로컬)"), "lmstudio": T("LM Studio (로컬)"), "mlx_server": T("mlx_lm.server (로컬)"),
            "llamacpp": T("llama.cpp server (로컬)"), "openai": "OpenAI", "anthropic": "Anthropic",
            "gemini": "Google Gemini", "openrouter": "OpenRouter", "groq": "Groq", "together": "Together AI",
            "deepseek": "DeepSeek", "custom": T("직접 입력")}.get(pid, pid)


def kind_label(kind: str) -> str:
    """화면 표시용 API 형식 이름 (특정 회사 이름을 쓰지 않는다)."""
    return {"auto": T("자동 감지 (권장)"), "openai": T("Chat Completions 형식"), "anthropic": T("Messages 형식"),
            "gemini": T("generateContent 형식")}.get(kind, kind)


KINDS = ("auto", "openai", "anthropic", "gemini")


def resolve_kind(kind: str, url: str) -> str:
    """'auto'면 주소로 형식을 정하고, 알 수 없으면 가장 널리 쓰이는 Chat Completions 형식으로 본다."""
    if kind == "auto" or kind not in ("openai", "anthropic", "gemini"):
        return guess_kind(url) or "openai"
    return kind

# 사용자가 붙여 넣은 주소 끝에 엔드포인트가 붙어 있으면 잘라낸다 (404의 가장 흔한 원인)
_ENDPOINT_SUFFIXES = ("/chat/completions", "/completions", "/responses", "/embeddings", "/v1/messages", "/messages",
                      "/models")


def guess_kind(url: str) -> str | None:
    """주소로 API 형식을 추정. 확실하지 않으면 None."""
    u = url.lower()
    if "anthropic.com" in u:
        return "anthropic"
    if "generativelanguage.googleapis.com" in u:
        return "openai" if "/openai" in u else "gemini"
    if any(h in u for h in ("api.openai.com", "openrouter.ai", "groq.com", "together.xyz", "deepseek.com", ":11434", ":1234")):
        return "openai"
    return None


def normalize_base_url(kind: str, url: str) -> str:
    """붙여 넣은 주소를 이 프로그램이 기대하는 '기본 주소' 형태로 정리한다.
    예) https://api.openai.com/v1/chat/completions → https://api.openai.com/v1
        https://api.openai.com                     → https://api.openai.com/v1
        http://localhost:11434                     → http://localhost:11434/v1
        https://api.anthropic.com/v1/messages      → https://api.anthropic.com
        .../v1beta/models/gemini-2.5-flash:generateContent → .../v1beta
    """
    u = (url or "").strip().strip('"').strip("'").rstrip("/")
    if not u:
        return u
    if "://" not in u:
        u = ("http://" if u.startswith(("localhost", "127.0.0.1")) else "https://") + u
    if "?" in u:
        u = u.split("?", 1)[0]
    if kind == "gemini" and ":generatecontent" in u.lower():
        u = u[: u.lower().rfind("/models/")]
    changed = True
    while changed:
        changed = False
        for suf in _ENDPOINT_SUFFIXES:
            if u.lower().endswith(suf):
                u = u[: -len(suf)].rstrip("/")
                changed = True
    path = urlparse(u).path.strip("/")
    if kind == "openai" and not path:
        u += "/v1"
    elif kind == "anthropic" and path.lower() == "v1":
        u = u[:-3]
    elif kind == "gemini" and not path:
        u += "/v1beta"
    return u


def is_local_url(url: str) -> bool:
    """localhost / 사설망 주소면 로컬로 판단(이용약관 경고 대상 아님)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False


def tos_url_for(base_url: str) -> str:
    for p in PRESETS.values():
        if p.base_url and p.tos_url and base_url.rstrip("/").startswith(p.base_url.rstrip("/")):
            return p.tos_url
    return ""


class LLMError(Exception):
    pass


@dataclass
class ProviderConfig:
    kind: str
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    concurrency: int = 2
    timeout: float = 600.0
    retries: int = 3


class _Adapted(Exception):
    """서버 오류 응답을 보고 요청 방식을 바꿨으니 다시 보내라는 내부 신호."""


class _StreamCollector:
    """스트리밍 조각을 모으며 작업 신호등에 토큰 신호를 보내고, 반복 출력(헛돎)을 감지한다."""

    def __init__(self, activity: Activity | None):
        self.activity = activity
        self.parts: list[str] = []
        self._n = 0

    def alive(self) -> None:
        """keep-alive·이벤트 등 바이트는 오지만 토큰은 아닌 경우: 연결만 살아 있음(신호로 치지 않음)."""

    def text(self, t: str) -> None:
        if not t:
            return
        self.parts.append(t)
        self._n += 1
        if self.activity:
            self.activity.beat(tokens=1)
        if self._n % 24 == 0 and is_looping("".join(self.parts[-400:])):
            if self.activity:
                self.activity.flag_loop()
            raise LLMError(T("반복 출력 감지(모델이 헛돌고 있음)"))

    def thinking(self, t: str) -> None:
        if t and self.activity:
            self.activity.beat(reasoning=1)

    def result(self) -> str:
        return "".join(self.parts)


async def _sse_events(r: httpx.Response, col: _StreamCollector):
    """SSE 'data:' 줄을 JSON으로 풀어 돌려준다."""
    async for line in r.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            col.alive()
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


class Provider:
    """공통 인터페이스. 하위 클래스는 _request / list_models 를 구현한다.
    기본은 스트리밍으로 받아 토큰마다 작업 신호를 보내고, 서버가 스트리밍을 거부하면 일반 요청으로 바꾼다."""

    _no_stream = False

    def __init__(self, cfg: ProviderConfig, transport: httpx.AsyncBaseTransport | None = None,
                 cancel: CancelToken | None = None, activity: Activity | None = None):
        self.cfg = cfg
        self.cancel = cancel
        self.activity = activity
        self._sem = asyncio.Semaphore(max(1, int(cfg.concurrency)))
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout, connect=15.0), transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    @property
    def base(self) -> str:
        return normalize_base_url(self.cfg.kind, self.cfg.base_url)

    async def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        """재시도/백오프/동시성 제한 포함 채팅 호출. <think> 블록은 제거해서 반환."""
        temp = self.cfg.temperature if temperature is None else temperature
        mtok = self.cfg.max_tokens if max_tokens is None else max_tokens
        last_err: Exception | None = None
        for attempt in range(self.cfg.retries + 1):
            if self.cancel:
                await self.cancel.checkpoint()
            try:
                async with self._sem:
                    if self.cancel:
                        self.cancel.check()
                    try:
                        text = await asyncio.wait_for(self._chat_once(messages, temp, mtok), self.cfg.timeout)
                    except asyncio.TimeoutError:
                        raise LLMError(T('응답 시간 초과({sec:.0f}초)', sec=self.cfg.timeout))
                return strip_thinking(text)
            except LLMError as e:
                last_err = e
                if not _retryable(e):
                    raise
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = LLMError(T('연결 오류: {e}', e=e))
            if attempt < self.cfg.retries:
                await asyncio.sleep(min(30.0, (2 ** attempt) + random.random()))
        raise last_err or LLMError(T('알 수 없는 오류'))

    async def chat_json(self, messages: list[dict[str, str]], *, temperature: float | None = None,
                        max_tokens: int | None = None, attempts: int = 3) -> Any:
        """JSON 응답을 기대하는 호출. 파싱 실패 시 오류 내용을 알려주고 다시 요청."""
        msgs = list(messages)
        last = ""
        for _ in range(attempts):
            last = await self.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            try:
                return extract_json(last)
            except ValueError:
                msgs = msgs + [
                    {"role": "assistant", "content": last[:4000]},
                    {"role": "user", "content": "Your response could not be parsed as JSON. Output only the requested JSON, with no explanation."},
                ]
        raise LLMError(T('JSON 응답 파싱 실패: ') + last[:300])

    async def quick_test(self, on_step=None, list_timeout: float = 10.0, chat_timeout: float = 45.0) -> dict:
        """연결 테스트: 재시도 없이 짧은 제한 시간으로 ① 주소·키(모델 목록) ② 모델의 짧은 응답을 확인한다."""
        act = self.activity
        if act:
            act.begin(T("주소·API 키 확인"))
        ok, err = False, ""
        self._in_quick_test = True  # 내부 요청은 같은 신호등을 이어서 쓴다(중복 집계 방지)
        try:
            out = await self._quick_test(on_step, list_timeout, chat_timeout)
            ok = True
            return out
        except Exception as e:
            err = str(e).split("\n")[0][:120]
            raise
        finally:
            self._in_quick_test = False
            if act:
                act.end(ok, err)

    async def _quick_test(self, on_step, list_timeout: float, chat_timeout: float) -> dict:
        import time

        out: dict[str, Any] = {"models_ok": None, "model_listed": None, "reply": "", "seconds": 0.0}
        if on_step:
            on_step(1)
        try:
            models = await asyncio.wait_for(self.list_models(), list_timeout)
            out["models_ok"] = True
            out["model_listed"] = (not models) or any(self.cfg.model == m or m.endswith("/" + self.cfg.model)
                                                     for m in models)
        except LLMError as e:
            if "HTTP 401" in str(e) or "HTTP 403" in str(e):
                raise
            out["models_ok"] = False  # 모델 목록 API가 없는 서버도 있으므로 계속 진행
        except (asyncio.TimeoutError, httpx.HTTPError):
            out["models_ok"] = False
        if self.activity:
            self.activity.beat(detail=T("모델 응답 확인"))
        if on_step:
            on_step(2)
        t0 = time.monotonic()
        try:
            text = await asyncio.wait_for(
                self._chat_once([{"role": "user", "content": "Reply with the single word: OK"}], 0.0, 32), chat_timeout)
        except asyncio.TimeoutError:
            raise LLMError(T('모델이 {sec:.0f}초 안에 응답하지 않았습니다. 주소·키는 정상일 수 있지만 서버가 붐비거나 추론(생각) 모델이라 느릴 수 있습니다.', sec=chat_timeout))
        except httpx.HTTPError as e:
            raise LLMError(T('연결 오류: {e}', e=e))
        out["reply"] = strip_thinking(text)
        out["seconds"] = time.monotonic() - t0
        return out

    async def _chat_once(self, messages, temperature, max_tokens) -> str:
        """한 번의 요청. 작업 신호등을 켜고(begin) 끝나면 성공/실패로 끈다(end)."""
        act = self.activity
        track = act is not None and not getattr(self, "_in_quick_test", False)
        if track:
            act.begin()
        ok, err = False, ""
        try:
            for _ in range(5):
                try:
                    text = await self._request(messages, temperature, max_tokens, _StreamCollector(act))
                    ok = True
                    return text
                except _Adapted:
                    continue
            raise LLMError(T('알 수 없는 오류'))
        except asyncio.CancelledError:
            err = T('중단됨')
            raise
        except Exception as e:
            err = str(e).split("\n")[0][:120]
            raise
        finally:
            if track:
                act.end(ok, err)

    async def _send(self, url: str, payload: dict, col: _StreamCollector, parse_stream, parse_json) -> str:
        """스트리밍(가능하면) 또는 일반 요청을 보내고 응답을 파싱한다."""
        stream = not self._no_stream
        body = dict(payload)
        if stream:
            body.update(self._stream_fields())
        async with self._client.stream("POST", url, json=body, headers=self._headers()) as r:
            if r.status_code >= 400:
                await r.aread()
                if self._adapt(r):
                    raise _Adapted()
                self._raise_for(r)
            ctype = r.headers.get("content-type", "")
            if stream and "text/event-stream" in ctype:
                await parse_stream(r, col)
                return col.result()
            # 일반 응답(또는 스트리밍을 무시한 서버): 바이트가 들어오는 것을 신호로 본다
            chunks = []
            async for b in r.aiter_bytes():
                chunks.append(b)
                if col.activity:
                    col.activity.beat()
            try:
                data = json.loads(b"".join(chunks) or b"{}")
            except json.JSONDecodeError as e:
                raise LLMError(T('예상하지 못한 응답 형식: {v}', v=b"".join(chunks)[:300])) from e
            return parse_json(data)

    def _stream_fields(self) -> dict:
        return {"stream": True}

    def _adapt(self, r: httpx.Response) -> bool:
        body = r.text.lower()
        if not self._no_stream and "stream" in body and ("not support" in body or "unsupported" in body):
            self._no_stream = True
            return True
        return False

    async def _request(self, messages, temperature, max_tokens, col: _StreamCollector) -> str:  # pragma: no cover
        raise NotImplementedError

    async def list_models(self) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def _raise_for(self, r: httpx.Response) -> None:
        if r.status_code < 400:
            return
        body = r.text[:500]
        try:
            url = str(r.request.url).split("?", 1)[0]
        except RuntimeError:
            url = ""
        hint = ""
        if r.status_code == 404:
            hint = T("주소 또는 모델명이 잘못되었을 수 있습니다. 요청한 주소: {url} · 확인: ① 주소는 엔드포인트를 뺀 기본 주소인지(예: https://서버주소/v1) ② API 형식이 서버와 맞는지(모르면 '자동 감지') ③ 모델명이 정확한지([모델 목록]으로 확인)", url=url)
        elif r.status_code in (401, 403):
            hint = T("API 키가 없거나 잘못되었거나 권한이 없습니다.")
        raise LLMError(f"HTTP {r.status_code}: {body}" + (f"\n→ {hint}" if hint else ""))


def _retryable(e: LLMError) -> bool:
    msg = str(e)
    return any(code in msg for code in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "HTTP 529",
                                        T('연결 오류'), T('응답 시간 초과({sec:.0f}초)', sec=0).split("(")[0],
                                        T("반복 출력 감지(모델이 헛돌고 있음)")))


class OpenAICompatibleProvider(Provider):
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    # 모델마다 다른 요구사항을 응답 오류를 보고 자동으로 맞춘다
    _use_mct = False  # max_tokens 대신 max_completion_tokens (OpenAI 최신 추론 모델)
    _no_temp = False  # temperature 미지원 모델
    _use_responses = False  # /responses 전용 모델

    async def _request(self, messages, temperature, max_tokens, col: _StreamCollector) -> str:
        if self._use_responses:
            return await self._send(f"{self.base}/responses", self._responses_payload(messages, temperature, max_tokens),
                                    col, self._parse_responses_stream, self._parse_responses_json)
        payload = {"model": self.cfg.model, "messages": messages,
                   ("max_completion_tokens" if self._use_mct else "max_tokens"): max_tokens}
        if not self._no_temp:
            payload["temperature"] = temperature
        return await self._send(f"{self.base}/chat/completions", payload, col, self._parse_stream, self._parse_json)

    @staticmethod
    async def _parse_stream(r: httpx.Response, col: _StreamCollector) -> None:
        async for obj in _sse_events(r, col):
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                col.thinking(d.get("reasoning_content") or d.get("reasoning") or "")
                c = d.get("content")
                if isinstance(c, list):
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                col.text(c or "")

    @staticmethod
    def _parse_json(data: dict) -> str:
        try:
            content = data["choices"][0]["message"].get("content") or ""
            if isinstance(content, list):  # 일부 서버는 파트 배열로 반환
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError(T('예상하지 못한 응답 형식: {v}', v=str(data)[:300])) from e

    @staticmethod
    async def _parse_responses_stream(r: httpx.Response, col: _StreamCollector) -> None:
        async for obj in _sse_events(r, col):
            t = obj.get("type", "")
            if t == "response.output_text.delta":
                col.text(obj.get("delta", ""))
            elif "reasoning" in t and t.endswith(".delta"):
                col.thinking(obj.get("delta", "") or "x")

    @staticmethod
    def _parse_responses_json(data: dict) -> str:
        return data.get("output_text") or "".join(
            c.get("text", "") for o in data.get("output", []) if o.get("type") == "message"
            for c in o.get("content", []) if isinstance(c, dict))

    def _adapt(self, r: httpx.Response) -> bool:
        body = r.text.lower()
        if not self._use_responses and r.status_code == 404 and ("v1/responses" in body or "responses api" in body):
            self._use_responses = True
            return True
        if not self._use_mct and "max_completion_tokens" in body:
            self._use_mct = True
            return True
        if not self._no_temp and "temperature" in body and ("unsupported" in body or "not support" in body
                                                            or "does not support" in body or "only the default" in body):
            self._no_temp = True
            return True
        return super()._adapt(r)

    def _responses_payload(self, messages, temperature, max_tokens) -> dict:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        p: dict[str, Any] = {"model": self.cfg.model, "max_output_tokens": max_tokens,
                             "input": [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]}
        if system:
            p["instructions"] = system
        if not self._no_temp:
            p["temperature"] = temperature
        return p

    async def list_models(self) -> list[str]:
        r = await self._client.get(f"{self.base}/models", headers=self._headers())
        self._raise_for(r)
        return sorted(m.get("id", "") for m in r.json().get("data", []) if m.get("id"))


class AnthropicProvider(Provider):
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.cfg.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}

    async def _request(self, messages, temperature, max_tokens, col: _StreamCollector) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        payload: dict[str, Any] = {"model": self.cfg.model, "messages": msgs, "max_tokens": max_tokens,
                                   "temperature": temperature}
        if system:
            payload["system"] = system
        return await self._send(f"{self.base}/v1/messages", payload, col, self._parse_stream, self._parse_json)

    @staticmethod
    async def _parse_stream(r: httpx.Response, col: _StreamCollector) -> None:
        async for obj in _sse_events(r, col):
            t = obj.get("type")
            if t == "content_block_delta":
                d = obj.get("delta") or {}
                if d.get("type") == "text_delta":
                    col.text(d.get("text", ""))
                elif d.get("type") == "thinking_delta":
                    col.thinking(d.get("thinking", "") or "x")
            elif t == "error":
                raise LLMError(f"Anthropic stream error: {str(obj)[:300]}")
            elif t == "message_stop":
                return

    @staticmethod
    def _parse_json(data: dict) -> str:
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    async def list_models(self) -> list[str]:
        r = await self._client.get(f"{self.base}/v1/models", headers=self._headers())
        self._raise_for(r)
        return sorted(m.get("id", "") for m in r.json().get("data", []) if m.get("id"))


class GeminiProvider(Provider):
    def _headers(self) -> dict[str, str]:
        # 키는 URL이 아니라 헤더로 전달
        return {"x-goog-api-key": self.cfg.api_key, "Content-Type": "application/json"}

    def _stream_fields(self) -> dict:
        return {}  # Gemini는 엔드포인트로 스트리밍을 구분한다

    async def _request(self, messages, temperature, max_tokens, col: _StreamCollector) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        payload: dict[str, Any] = {"contents": contents,
                                   "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        model = self.cfg.model if self.cfg.model.startswith("models/") else f"models/{self.cfg.model}"
        url = (f"{self.base}/{model}:generateContent" if self._no_stream
               else f"{self.base}/{model}:streamGenerateContent?alt=sse")
        return await self._send(url, payload, col, self._parse_stream, self._parse_json)

    @staticmethod
    async def _parse_stream(r: httpx.Response, col: _StreamCollector) -> None:
        async for obj in _sse_events(r, col):
            for cand in obj.get("candidates") or []:
                for p in (cand.get("content") or {}).get("parts") or []:
                    if p.get("thought"):
                        col.thinking(p.get("text", "") or "x")
                    else:
                        col.text(p.get("text", ""))

    @staticmethod
    def _parse_json(data: dict) -> str:
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(T('예상하지 못한 응답 형식: {v}', v=str(data)[:300])) from e

    async def list_models(self) -> list[str]:
        r = await self._client.get(f"{self.base}/models", headers=self._headers())
        self._raise_for(r)
        return sorted(m.get("name", "").removeprefix("models/") for m in r.json().get("models", []))


def make_provider(cfg: ProviderConfig, transport: httpx.AsyncBaseTransport | None = None,
                  cancel: CancelToken | None = None, require_model: bool = True,
                  activity: Activity | None = None) -> Provider:
    if not cfg.base_url:
        raise ValueError(T('API 주소가 비어 있습니다.'))
    if cfg.kind == "auto":
        import dataclasses

        cfg = dataclasses.replace(cfg, kind=resolve_kind("auto", cfg.base_url))
    cls = {"openai": OpenAICompatibleProvider, "anthropic": AnthropicProvider, "gemini": GeminiProvider}.get(cfg.kind)
    if cls is None:
        raise ValueError(T('지원하지 않는 API 형식: {kind}', kind=cfg.kind))
    g = guess_kind(cfg.base_url)
    if g and g != cfg.kind:
        import dataclasses

        cfg = dataclasses.replace(cfg, kind=g)  # 주소와 맞지 않는 API 형식은 자동 보정
        cls = {"openai": OpenAICompatibleProvider, "anthropic": AnthropicProvider, "gemini": GeminiProvider}[g]
    if require_model and not cfg.model:
        raise ValueError(T('모델명이 비어 있습니다.'))
    return cls(cfg, transport=transport, cancel=cancel, activity=activity)


def provider_config_from_role(role) -> ProviderConfig:
    """config.RoleConfig → ProviderConfig"""
    return ProviderConfig(kind=role.kind, base_url=role.base_url, model=role.model, api_key=role.api_key,
                          temperature=role.temperature, max_tokens=role.max_tokens, concurrency=role.concurrency)
