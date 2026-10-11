"""단위 테스트: python -m unittest discover -s tests"""
import asyncio
import json
import random
import unittest

import httpx

from app.core import algorithm, hardware
from app.core.jsonutil import extract_json
from app.core.providers import ProviderConfig, is_local_url, make_provider
from app.core.quality import Deduper, decide, judge_pair
from app.core.roles import _difficulty_plan, split_summary, validate_spec

QWEN05 = {"model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24, "intermediate_size": 4864,
          "num_attention_heads": 14, "num_key_value_heads": 2, "vocab_size": 151936, "tie_word_embeddings": True}


class TestHardware(unittest.TestCase):
    def test_param_estimate_qwen05(self):
        m = hardware.model_info_from_config("Qwen/Qwen2.5-0.5B-Instruct", QWEN05)
        self.assertAlmostEqual(m.params_b, 0.49, delta=0.03)

    def test_check_insufficient_and_ok(self):
        m = hardware.model_info_from_config("x", QWEN05)
        small = hardware.HardwareInfo("M1", "arm64", "15.0", 8, 0.5, 100)
        r = hardware.check(small, m, "mlx", "auto", 1, 512, 16)
        self.assertFalse(r.ok)
        self.assertTrue(r.messages)
        big = hardware.HardwareInfo("M3 Max", "arm64", "15.0", 64, 48, 500)
        r = hardware.check(big, m, "mlx", "auto", 1, 512, 16)
        self.assertTrue(r.ok)
        self.assertIn("lora", r.feasible_modes)
        self.assertEqual(r.feasible_modes[0], "qlora")  # 가벼운 순

    def test_explicit_mode_not_auto_switched(self):
        m = hardware.model_info_from_config("x", {**QWEN05, "hidden_size": 4096, "num_hidden_layers": 32,
                                                  "intermediate_size": 14336, "num_attention_heads": 32,
                                                  "num_key_value_heads": 8, "tie_word_embeddings": False})
        hw = hardware.HardwareInfo("M2", "arm64", "15.0", 24, 16, 500)
        r = hardware.check(hw, m, "mlx", "full", 1, 512, 16)
        self.assertFalse(r.ok)  # 8B 전체 파인튜닝은 16GB로 불가, LoRA로 자동 전환하지 않음
        self.assertIn("qlora", r.feasible_modes)

    def test_disk_check(self):
        m = hardware.model_info_from_config("x", QWEN05)
        hw = hardware.HardwareInfo("M1", "arm64", "15.0", 16, 12, 0.5)
        self.assertFalse(hardware.check(hw, m, "mlx", "auto", 1, 512, 16).ok)


class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        self.m = hardware.model_info_from_config("x", QWEN05)
        hw = hardware.HardwareInfo("M3", "arm64", "15.0", 64, 48, 500)
        self.r = hardware.check(hw, self.m, "mlx", "auto", 1, 512, 16)

    def test_rank_grows_with_data(self):
        r_small, _ = algorithm.choose_rank(self.m, 10_000)
        r_big, _ = algorithm.choose_rank(self.m, 500_000_000)
        self.assertLessEqual(r_small, r_big)

    def test_knowledge_goal_prefers_full(self):
        rec = algorithm.recommend("knowledge", 300, 300, self.m, self.r, "mlx")
        self.assertEqual(rec.config.mode, "full")
        self.assertFalse(rec.config.lr_sweep)
        self.assertAlmostEqual(rec.config.learning_rate, algorithm.FULL_FT_BASE_LR)

    def test_adaptation_lora_lr_x10(self):
        rec = algorithm.recommend("adaptation", 2000, 300, self.m, self.r, "mlx")
        self.assertIn(rec.config.mode, ("lora", "rslora"))
        self.assertAlmostEqual(rec.config.learning_rate, algorithm.FULL_FT_BASE_LR * algorithm.LORA_LR_MULT)
        self.assertLess(rec.config.batch_size, 32)

    def test_overrides(self):
        rec = algorithm.recommend("adaptation", 100, 300, self.m, self.r, "mlx",
                                  {"mode": "dora", "rank": 8, "learning_rate": 3e-4, "batch_size": 2})
        c = rec.config
        self.assertEqual((c.mode, c.rank, c.learning_rate, c.batch_size), ("dora", 8, 3e-4, 2))

    def test_rslora_scale(self):
        c = algorithm.TrainConfig(mode="rslora", rank=64, alpha=32)
        self.assertAlmostEqual(c.scale, 32 / 8)


class TestQuality(unittest.TestCase):
    def test_dedup(self):
        d = Deduper(0.85)
        self.assertTrue(d.add("파이썬에서 리스트를 정렬하는 방법은?"))
        self.assertFalse(d.add("파이썬에서 리스트를 정렬하는 방법은?!"))
        self.assertTrue(d.add("자바스크립트 배열의 길이는 어떻게 구하나요?"))

    def test_decide_rules(self):
        self.assertEqual(decide(8, True, 9, 5, False, 7, True).chosen, "1")          # ① 한쪽만 통과
        v = decide(8, True, 8, 9, False, 7, True)
        self.assertEqual((v.status, v.chosen), ("accepted", "2"))                     # ② 둘 다 통과
        self.assertEqual(decide(8, True, 8, 9, True, 7, True).status, "regenerate")    # ③ 모순 → 재생성
        self.assertEqual(decide(8, True, 8, 9, True, 7, False).status, "uncertain")    # ③ 재생성 불가 → 제외
        self.assertEqual(decide(8, True, 3, 4, False, 7, True).status, "regenerate")   # ④ 둘 다 미달
        self.assertEqual(decide(8, True, 3, 4, False, 7, False).status, "rejected")
        self.assertEqual(decide(2, True, 9, 9, False, 7, True).status, "rejected_question")  # ⑤
        self.assertFalse(decide(8, True, 8, 8.5, False, 7, True, swap_flipped=True).clear_winner)

    def test_score_fallback_when_overall_missing(self):
        from app.core.quality import _score
        self.assertEqual(_score({"accuracy": 8, "completeness": 6, "format": 10, "safety": 10}), 8.5)
        self.assertEqual(_score({"overall": 7}), 7)

    def test_judge_pair_blind_and_swap(self):
        class FakeOrch:
            def __init__(self):
                self.calls = []

            async def judge_pair(self, plan, spec, item, a, b):
                self.calls.append((a, b))
                # 항상 첫 번째 답에 약간 높은 점수 → 위치 편향 → 순서 바꾸면 승자 뒤집힘
                return {"question": {"overall": 9, "difficulty_ok": True}, "A": {"overall": 8.5},
                        "B": {"overall": 8.0}, "contradiction": False}

        o = FakeOrch()
        v = asyncio.run(judge_pair(o, {}, {}, {"question": "q"}, "ans1", "ans2", 7, True, random.Random(0)))
        self.assertEqual(len(o.calls), 2)              # 점수 차가 작아 순서 교체 재평가
        self.assertEqual(o.calls[0], o.calls[1][::-1])
        self.assertEqual(v.status, "accepted")
        self.assertFalse(v.clear_winner)                # 승자 뒤집힘 → 선호도 쌍 저장 안 함


class TestJsonAndRoles(unittest.TestCase):
    def test_extract_json(self):
        self.assertEqual(extract_json('<think>hmm {"a":0}</think>\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('결과: {"items": [{"q": "x}"}]} 끝'), {"items": [{"q": "x}"}]})
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_split_summary(self):
        body, summ = split_summary("질문입니다\n<plan_summary>- 목표: A</plan_summary>")
        self.assertEqual(body, "질문입니다")
        self.assertIn("목표", summ)

    def test_validate_spec_scales_to_target(self):
        spec = validate_spec({"categories": [{"name": "a", "count": 10}, {"name": "b", "count": 30}]},
                             {"target_samples": 80})
        self.assertEqual(sum(c["count"] for c in spec["categories"]), 80)

    def test_spec_placeholder_rejected(self):
        bad = {"system_prompt": "<a short system prompt shared by all student training examples>",
               "categories": [{"name": "지리", "count": 5}]}
        with self.assertRaises(ValueError):
            validate_spec(bad, {"target_samples": 5})
        self.assertEqual(validate_spec(bad, {"target_samples": 5}, strict=False)["system_prompt"], "")

    def test_validate_spec_exact_total(self):
        spec = validate_spec({"categories": [{"name": n, "count": 5} for n in "abc"]}, {"target_samples": 16})
        self.assertEqual(sum(c["count"] for c in spec["categories"]), 16)

    def test_difficulty_plan(self):
        self.assertEqual(sum(_difficulty_plan(7, {"easy": 0.3, "medium": 0.5, "hard": 0.2}).values()), 7)


class TestProviders(unittest.TestCase):
    def test_normalize_pasted_urls(self):
        from app.core.providers import guess_kind, normalize_base_url as n
        self.assertEqual(n("openai", "https://api.openai.com/v1/chat/completions"), "https://api.openai.com/v1")
        self.assertEqual(n("openai", "https://api.openai.com/"), "https://api.openai.com/v1")
        self.assertEqual(n("openai", "localhost:11434"), "http://localhost:11434/v1")
        self.assertEqual(n("openai", "https://openrouter.ai/api/v1/"), "https://openrouter.ai/api/v1")
        self.assertEqual(n("anthropic", "https://api.anthropic.com/v1/messages"), "https://api.anthropic.com")
        self.assertEqual(n("gemini", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=x"),
                         "https://generativelanguage.googleapis.com/v1beta")
        self.assertEqual(guess_kind("https://api.anthropic.com/v1"), "anthropic")
        self.assertEqual(guess_kind("https://generativelanguage.googleapis.com/v1beta/openai"), "openai")

    def test_kind_autocorrect_and_responses_fallback(self):
        calls = []

        def h(req):
            calls.append(str(req.url))
            if req.url.path.endswith("/chat/completions"):
                return httpx.Response(404, json={"error": {"message": "This model is only supported in v1/responses"}})
            return httpx.Response(200, json={"output_text": "pong"})

        self.assertEqual(self._run("openai", "https://api.openai.com/v1/chat/completions", h), "pong")
        self.assertTrue(calls[-1].endswith("/v1/responses"))

        def h2(req):
            self.assertEqual(req.url.path, "/v1/messages")  # OpenAI 형식으로 골라도 Anthropic 주소면 자동 보정
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        self.assertEqual(self._run("openai", "https://api.anthropic.com/v1", h2), "ok")

    def test_auto_kind(self):
        from app.core.providers import AnthropicProvider, GeminiProvider, OpenAICompatibleProvider, resolve_kind
        self.assertEqual(resolve_kind("auto", "http://localhost:11434/v1"), "openai")
        self.assertEqual(resolve_kind("auto", "https://my-gateway.example.com/v1"), "openai")
        for url, cls in (("https://api.anthropic.com", AnthropicProvider),
                         ("https://generativelanguage.googleapis.com/v1beta", GeminiProvider),
                         ("http://127.0.0.1:8080/v1", OpenAICompatibleProvider)):
            p = make_provider(ProviderConfig(kind="auto", base_url=url, model="m"))
            self.assertIsInstance(p, cls)
            asyncio.run(p.aclose())

    def test_404_message_has_hint(self):
        from app.core.providers import LLMError
        with self.assertRaises(LLMError) as cm:
            self._run("openai", "http://x/v1", lambda req: httpx.Response(404, text="not found"), retries=0)
        self.assertIn("http://x/v1/chat/completions", str(cm.exception))

    def test_quick_test_is_fast_when_model_hangs(self):
        import time
        from app.core.providers import LLMError

        async def h(req):
            if req.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "m"}]})
            await asyncio.sleep(30)  # 응답 없이 매달린 서버
            return httpx.Response(200, json={})

        async def go():
            p = make_provider(ProviderConfig(kind="openai", base_url="https://api.deepseek.com", model="m"),
                              transport=httpx.MockTransport(h))
            try:
                return await p.quick_test(chat_timeout=0.5)
            finally:
                await p.aclose()

        t0 = time.monotonic()
        with self.assertRaises(LLMError):
            asyncio.run(go())
        self.assertLess(time.monotonic() - t0, 5)

    def test_quick_test_success_and_unlisted_model(self):
        def h(req):
            if req.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "other"}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

        async def go():
            p = make_provider(ProviderConfig(kind="openai", base_url="https://api.deepseek.com", model="m"),
                              transport=httpx.MockTransport(h))
            try:
                return await p.quick_test()
            finally:
                await p.aclose()

        r = asyncio.run(go())
        self.assertEqual(r["reply"], "OK")
        self.assertFalse(r["model_listed"])

    def test_chat_has_total_deadline(self):
        from app.core.providers import LLMError

        async def h(req):
            await asyncio.sleep(30)
            return httpx.Response(200, json={})

        async def go():
            p = make_provider(ProviderConfig(kind="openai", base_url="http://x/v1", model="m", timeout=0.3, retries=0),
                              transport=httpx.MockTransport(h))
            try:
                return await p.chat([{"role": "user", "content": "hi"}])
            finally:
                await p.aclose()

        with self.assertRaises(LLMError):
            asyncio.run(go())

    def test_local_detection(self):
        self.assertTrue(is_local_url("http://localhost:11434/v1"))
        self.assertTrue(is_local_url("http://192.168.0.10:8080/v1"))
        self.assertFalse(is_local_url("https://api.openai.com/v1"))

    def _run(self, kind, base, handler, **kw):
        async def go():
            p = make_provider(ProviderConfig(kind=kind, base_url=base, model="m", api_key="sk-test", **kw),
                              transport=httpx.MockTransport(handler))
            try:
                return await p.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
            finally:
                await p.aclose()
        return asyncio.run(go())

    def test_openai_retry_429(self):
        calls = {"n": 0}

        def h(req):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": "rate"})
            self.assertEqual(req.headers["authorization"], "Bearer sk-test")
            return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think>hello"}}]})

        self.assertEqual(self._run("openai", "http://x/v1", h, retries=2), "hello")
        self.assertEqual(calls["n"], 2)

    def test_anthropic(self):
        def h(req):
            body = json.loads(req.content)
            self.assertEqual(body["system"], "s")
            self.assertEqual(req.headers["x-api-key"], "sk-test")
            return httpx.Response(200, json={"content": [{"type": "text", "text": "hey"}]})

        self.assertEqual(self._run("anthropic", "https://api.anthropic.com", h), "hey")

    def test_gemini_key_in_header_not_url(self):
        def h(req):
            self.assertNotIn("key=", str(req.url))
            self.assertEqual(req.headers["x-goog-api-key"], "sk-test")
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "yo"}]}}]})

        self.assertEqual(self._run("gemini", "https://g/v1beta", h), "yo")


class TestActivityLights(unittest.TestCase):
    def test_states(self):
        import time
        from app.core.activity import GRAY, GREEN, ORANGE, RED, Activity
        a = Activity("x", "x", first_grace=10, stall_after=5)
        self.assertEqual(a.evaluate().color, GRAY)
        a.begin()
        t0 = a.started
        self.assertEqual(a.evaluate(t0 + 3).color, ORANGE)       # 첫 토큰 대기 중
        self.assertEqual(a.evaluate(t0 + 11).color, RED)         # 첫 신호가 너무 늦음
        a.beat(reasoning=1)
        b = a.last_beat
        st = a.evaluate(b + 1)
        self.assertEqual(st.color, ORANGE)                        # 추론 중
        self.assertEqual(a.evaluate(b + 6).color, RED)           # 신호 끊김 → 멈춤
        a.beat(tokens=1)
        a.flag_loop()
        self.assertEqual(a.evaluate(time.monotonic()).color, RED) # 헛돎
        a.end(True)
        self.assertEqual(a.evaluate().color, GREEN)
        a.begin(); a.end(False, "boom")
        self.assertEqual(a.evaluate().color, RED)

    def test_is_looping(self):
        from app.core.activity import is_looping
        self.assertTrue(is_looping("정상 시작. " + "같은 말을 반복합니다. " * 8))
        self.assertFalse(is_looping("서울은 대한민국의 수도이며 인구가 가장 많은 도시입니다. 부산은 제2의 도시입니다."))

    def _stream(self, kind, base, sse_lines, act):
        body = "".join(l + "\n" for l in sse_lines).encode()

        def h(req):
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        async def go():
            p = make_provider(ProviderConfig(kind=kind, base_url=base, model="m", retries=0),
                              transport=httpx.MockTransport(h), activity=act)
            try:
                return await p.chat([{"role": "user", "content": "hi"}])
            finally:
                await p.aclose()
        return asyncio.run(go())

    def test_openai_stream_with_reasoning(self):
        from app.core.activity import GREEN, Activity
        a = Activity("o", "o")
        lines = [": keep-alive", 'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}',
                 'data: {"choices":[{"delta":{"content":"Hel"}}]}', 'data: {"choices":[{"delta":{"content":"lo"}}]}',
                 "data: [DONE]"]
        self.assertEqual(self._stream("openai", "https://api.deepseek.com", lines, a), "Hello")
        self.assertEqual((a.tokens, a.reasoning), (2, 1))
        self.assertEqual(a.evaluate().color, GREEN)

    def test_anthropic_and_gemini_stream(self):
        from app.core.activity import Activity
        an = ['event: ping', 'data: {"type":"ping"}',
              'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"..."}}',
              'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"안녕"}}',
              'data: {"type":"message_stop"}']
        self.assertEqual(self._stream("anthropic", "https://api.anthropic.com", an, Activity("a", "a")), "안녕")
        ge = ['data: {"candidates":[{"content":{"parts":[{"text":"think","thought":true}]}}]}',
              'data: {"candidates":[{"content":{"parts":[{"text":"答え"}]}}]}']
        self.assertEqual(self._stream("gemini", "https://generativelanguage.googleapis.com/v1beta", ge, Activity("g", "g")), "答え")

    def test_stream_loop_is_aborted(self):
        from app.core.activity import RED, Activity
        from app.core.providers import LLMError
        a = Activity("l", "l")
        lines = ['data: {"choices":[{"delta":{"content":"반복 문장. "}}]}'] * 60 + ["data: [DONE]"]
        with self.assertRaises(LLMError):
            self._stream("openai", "http://x/v1", lines, a)
        self.assertEqual(a.evaluate().color, RED)


class TestEnglishPlanAndStorage(unittest.TestCase):
    def test_english_detection(self):
        from app.core.roles import non_english_ratio, plan_is_english
        self.assertLess(non_english_ratio("Teach two-digit addition. Café naïve"), 0.01)
        self.assertGreater(non_english_ratio("두 자리 수 덧셈을 가르친다"), 0.9)
        self.assertFalse(plan_is_english({"goal": "덧셈을 정확히", "title": "Addition"}))
        self.assertTrue(plan_is_english({"goal": "Add correctly", "domain": "Math", "language": "Korean",
                                         "user_summary_markdown": "한국어 요약은 검사하지 않음"}))

    def test_ensure_english_translates(self):
        from app.core.roles import Orchestrator

        class FakeProvider:
            async def chat_json(self, msgs, **kw):
                assert "Translate the values" in msgs[0]["content"]
                return {"title": "Addition teacher", "goal": "Answer addition correctly"}

        plan = {"title": "덧셈 선생님", "goal": "덧셈에 정확히 답한다", "language": "Korean"}
        out = asyncio.run(Orchestrator(FakeProvider()).ensure_english(plan))
        self.assertEqual(out["goal"], "Answer addition correctly")
        self.assertEqual(out["language"], "Korean")

    def test_project_layout_reset_and_flat_migration(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        from app.core import paths
        from app.core.project import F_PLAN, F_SPEC, F_T1, F_TRAIN, Project
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "p"
            root.mkdir()
            (root / "project.json").write_text('{"plan": {"goal": "x"}, "baseline": {"q1": "a"}, "train": {}}')
            (root / "01_learning_plan.md").write_text("plan")   # 이전 버전의 평평한 구조
            (root / "05_teacher1.jsonl").write_text("{}")
            (root / "adapters").mkdir()
            p = Project(root)
            self.assertTrue((root / F_PLAN).exists() and (root / F_T1).exists())
            self.assertTrue((root / "4_training" / "adapters").is_dir())
            p.path(F_SPEC).write_text("{}")
            p.path(F_TRAIN).write_text("{}")
            trashed = []
            with mock.patch.object(paths, "move_to_trash", side_effect=lambda x: trashed.append(Path(x).name) or
                                   __import__("shutil").rmtree(x) if Path(x).is_dir() else Path(x).unlink()):
                p.reset_from("generated")
            self.assertTrue((root / F_SPEC).exists())                 # 명세는 유지
            self.assertFalse((root / F_T1).exists())                  # 생성 데이터부터 삭제
            self.assertFalse((root / F_TRAIN).exists())
            st = p.state()
            self.assertNotIn("baseline", st)
            self.assertEqual(st["status"], "planned")

    def test_migrate_legacy_workspace(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        from app.core import paths
        with tempfile.TemporaryDirectory() as d:
            legacy = Path(d) / "repo_ws"
            (legacy / "old").mkdir(parents=True)
            (legacy / "old" / "project.json").write_text("{}")
            import os
            env = {k: v for k, v in os.environ.items() if k != "ESLLM_DATA"}
            with mock.patch.object(paths, "LEGACY_WORKSPACE", legacy), \
                    mock.patch.object(paths, "LEGACY_MODELS", Path(d) / "nomodels"), \
                    mock.patch.dict(os.environ, env, clear=True):
                old_root = paths.DATA_ROOT
                paths.set_data_root(Path(d) / "data")
                try:
                    moved = paths.migrate_legacy()
                    self.assertTrue((Path(d) / "data" / "projects" / "old" / "project.json").exists())
                    self.assertEqual(len(moved), 1)
                finally:
                    paths.set_data_root(old_root)


class TestI18n(unittest.TestCase):
    def test_all_ui_strings_translated(self):
        import ast
        import re
        from pathlib import Path
        from app.locales.en import MESSAGES as EN
        from app.locales.ja import MESSAGES as JA
        ph = lambda s: sorted(set(re.findall(r"(?<!\{)\{([A-Za-z_]\w*)", s)))
        root = Path(__file__).resolve().parents[1] / "app"
        missing = []
        for f in root.rglob("*.py"):
            for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "T" and n.args and isinstance(n.args[0], ast.Constant):
                    k = n.args[0].value
                    for name, cat in (("en", EN), ("ja", JA)):
                        if k not in cat:
                            missing.append(f"{name}: {f.name}: {k}")
                        elif ph(k) != ph(cat[k]):
                            missing.append(f"{name} placeholder mismatch: {k}")
        self.assertEqual(missing, [])

    def test_translate_with_format(self):
        from app import i18n
        old = i18n.language()
        try:
            i18n.set_language("ja")
            self.assertEqual(i18n.T("모델 {len}개", len=3), "モデル 3 個")
            i18n.set_language("en")
            self.assertEqual(i18n.T("저장"), "Save")
        finally:
            i18n.set_language(old)


if __name__ == "__main__":
    unittest.main()
