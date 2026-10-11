"""전체 파이프라인: 데이터 명세 → 사양 검사 → 교사1·교사2·평가(병렬 스트리밍) → 학생 학습 → 평가.

병렬 구조
- 교사1(질문+답변1) → 큐 → 교사2(질문만 보고 독립 답변2) → 큐 → 오케스트레이터 비교 평가
- 사양 검사 통과 직후 학생 모델 다운로드가 생성 단계와 동시에 진행
- 메모리 여유가 있으면 학생 모델의 학습 전 기준 응답도 생성 단계와 동시에 생성
"""
from __future__ import annotations

from ..i18n import T

import asyncio
import dataclasses
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Awaitable, Callable

from . import algorithm, hardware
from .activity import HUB
from .cancel import Cancelled, CancelToken
from .config import ROLES, AppConfig
from .events import FileLogReporter, Reporter
from .jsonutil import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from .local_backends import (StudentServer, ensure_ollama, is_ollama_url, ollama_models, ollama_root,
                             ollama_unload)
from .proc import KeepAwake
from .project import (D_DATA, D_TRAIN, F_PIPE_LOG, F_PLAN, F_ALGO, F_ALGO_DRAFT, F_EVAL, F_HW, F_JUDGE_LOG, F_PREF, F_PROV, F_QUALITY, F_SPEC, F_T1,
                      F_T2, F_TRAIN, F_TRAIN_LOG, F_VALID, Project)
from .provenance import write_provenance
from .providers import LLMError, OpenAICompatibleProvider, ProviderConfig, make_provider, provider_config_from_role
from .quality import Deduper, approx_tokens, judge_pair, length_ok
from .roles import Orchestrator, Teacher1, Teacher2, plan_is_english
from .trainers.base import TrainProgress, download_model
from .trainers.registry import get_engine, pick_auto_engine

AskUser = Callable[[str, str], Awaitable[bool]]
TERMINAL = {"accepted", "rejected", "rejected_question", "uncertain", "error"}
MAX_CONSECUTIVE_ERRORS = 6


# 파이프라인 신호등: 역할/작업별 하나씩 (UI가 같은 키로 찾아 그린다)
PIPELINE_ACTIVITIES = ("orchestrator", "teacher1", "teacher2", "download", "student", "train")


def pipeline_activity(key: str):
    labels = {"orchestrator": T("오케스트레이터"), "teacher1": T("교사1"), "teacher2": T("교사2"),
              "download": T("학생 모델 다운로드"), "student": T("학생 모델 (로딩·응답)"), "train": T("학생 학습")}
    # 학습·다운로드는 단계 사이 정적 구간(체크포인트 저장 등)이 길 수 있어 여유를 더 준다
    kw = {"stall_after": 90.0, "first_grace": 180.0} if key in ("train", "download", "student") else {}
    return HUB.get(f"pipe:{key}", labels.get(key, key), **kw)


class SpecInsufficientError(Cancelled):
    """컴퓨터 사양이 학생 모델 기동·학습에 부족 → 모든 작업 취소."""

    def __init__(self, report: hardware.HardwareReport):
        super().__init__(T('컴퓨터 사양 부족'))
        self.report = report


class _ErrorGuard:
    """역할별 연속 실패 감시: 계속 실패하면 파이프라인을 멈춘다(빈 데이터로 진행하지 않도록)."""

    def __init__(self, role: str):
        self.role = role
        self.count = 0

    def ok(self) -> None:
        self.count = 0

    def fail(self, e: Exception) -> None:
        self.count += 1
        if self.count >= MAX_CONSECUTIVE_ERRORS:
            raise RuntimeError(T('{role} 호출이 {count}번 연속 실패했습니다. 설정(주소/키/모델명)을 확인하세요. 마지막 오류: {e}', role=self.role, count=self.count, e=e))


class Pipeline:
    def __init__(self, project: Project, cfg: AppConfig, reporter: Reporter, cancel: CancelToken,
                 ask_user: AskUser | None = None):
        self.project = project
        self.cfg = cfg
        self.rep: Reporter = FileLogReporter(reporter, project.path(F_PIPE_LOG))
        self.cancel = cancel
        self.ask_user = ask_user
        self.rng = random.Random(1234)
        self.providers: list = []
        self.engine_id = cfg.student.engine
        self.model_info: hardware.ModelInfo | None = None
        self.report: hardware.HardwareReport | None = None
        self.model_path: Path | None = None
        self.download_task: asyncio.Task | None = None
        self.baseline_task: asyncio.Task | None = None
        self.accepted: list[dict] = []
        self.gen_done = asyncio.Event()

    # ------------------------------------------------------------------ 공통
    def log(self, msg: str, level: str = "INFO") -> None:
        self.rep.log(level, msg)

    def _provider(self, role: str, concurrency: int | None = None):
        pc = provider_config_from_role(self.cfg.roles[role])
        if concurrency is not None:
            pc.concurrency = concurrency
        p = make_provider(pc, cancel=self.cancel, activity=pipeline_activity(role))
        self.providers.append(p)
        return p

    def _concurrency(self, role: str) -> int:
        """같은 로컬 백엔드를 여러 역할이 공유하면 역할별 동시성을 자동으로 줄인다."""
        rc = self.cfg.roles[role]
        from .providers import is_local_url

        if not is_local_url(rc.base_url):
            return max(1, rc.concurrency)
        shared = sum(1 for r in ROLES if self.cfg.roles[r].base_url.rstrip("/") == rc.base_url.rstrip("/"))
        return max(1, rc.concurrency // shared) if shared > 1 else max(1, rc.concurrency)

    async def run(self) -> None:
        for key in PIPELINE_ACTIVITIES:
            pipeline_activity(key).reset()
        with KeepAwake():
            try:
                await self._run()
            finally:
                for t in (self.download_task, self.baseline_task):
                    if t and not t.done():
                        t.cancel()
                for p in self.providers:
                    try:
                        await p.aclose()
                    except Exception:
                        pass

    async def _run(self) -> None:
        plan = self.project.plan
        if not plan:
            raise RuntimeError(T('학습방안이 확정되지 않았습니다. [학습방안] 탭에서 먼저 방안을 확정하세요.'))
        self.project.update_state(status="running")
        await self._ensure_local_backends()
        for r in ROLES:
            if not self.cfg.roles[r].model:
                raise RuntimeError(T('{r} 모델명이 비어 있습니다. 설정 탭을 확인하세요.', r=r))
        self.orch = Orchestrator(self._provider("orchestrator", self._concurrency("orchestrator")))
        self.t1 = Teacher1(self._provider("teacher1", self._concurrency("teacher1")))
        self.t2 = Teacher2(self._provider("teacher2", self._concurrency("teacher2")))
        if not plan_is_english(plan):  # 이전 버전에서 만든 방안: 교사에게 넘기기 전에 영어로 변환
            self.log(T("학습방안을 영어로 변환합니다 (교사에게는 영어 방안만 전달)."))
            plan = await self.orch.ensure_english(plan)
            self.project.update_state(plan=plan)
            from .planning import plan_markdown

            self.project.path(F_PLAN).write_text(plan_markdown(plan), encoding="utf-8")

        spec = await self.stage_spec(plan)
        await self.stage_hwcheck(plan, spec)
        self.download_task = asyncio.create_task(self.stage_download())
        if self._baseline_parallel_ok():
            self.baseline_task = asyncio.create_task(self.stage_baseline(spec, wait_for_data=True))
        else:
            self.log(T('메모리 여유가 부족하거나 옵션이 꺼져 있어, 학습 전 기준 응답은 데이터 생성 후에 만듭니다.'))
        await self.stage_generate(plan, spec)
        self.gen_done.set()

        self.model_path = await self.download_task
        if self.baseline_task:
            await self.baseline_task
        await self._unload_local_models()
        if not self.baseline_task:
            await self.stage_baseline(spec, wait_for_data=False)
        await self.stage_train_and_eval(plan, spec)
        self.project.update_state(status="done")
        self.log(T('모든 단계가 완료되었습니다. [결과] 탭에서 확인하세요.'))

    async def _ensure_local_backends(self) -> None:
        for r in ROLES:
            rc = self.cfg.roles[r]
            if is_ollama_url(rc.base_url):
                root = ollama_root(rc.base_url)
                if not await ensure_ollama(root, self.rep):
                    raise RuntimeError(T('Ollama 서버를 시작할 수 없습니다. 설정 탭에서 Ollama를 설치/실행하세요.'))
                models = await ollama_models(root)
                from .local_backends import ollama_has

                if not ollama_has(models, rc.model):
                    raise RuntimeError(T("Ollama에 '{model}' 모델이 없습니다. 설정 탭에서 [모델 받기]를 누르세요.", model=rc.model))

    # ------------------------------------------------------------------ 1. 데이터 명세
    async def stage_spec(self, plan: dict) -> dict:
        p = self.project.path(F_SPEC)
        if p.exists():
            self.rep.stage("spec", 1, 1, T('기존 명세 사용'))
            return read_json(p)
        self.rep.stage("spec", 0, 1, T('오케스트레이터가 명세 작성 중'))
        self.log(T('오케스트레이터: 학습데이터 명세 작성'))
        spec = await self.orch.make_data_spec(plan)
        write_json(p, spec)
        total = sum(c["count"] for c in spec["categories"])
        self.log(T('명세 완료: 카테고리 {len}개, 목표 {total}문항', len=len(spec['categories']), total=total))
        self.rep.stage("spec", 1, 1, T('{len}개 카테고리 / {total}문항', len=len(spec['categories']), total=total))
        return spec

    # ------------------------------------------------------------------ 2. 사양 검사 (교사1 단계 시작 전)
    async def stage_hwcheck(self, plan: dict, spec: dict) -> None:
        self.rep.stage("hwcheck", 0, 1, T('사양 측정 중'))
        s = self.cfg.student
        try:
            mcfg = await asyncio.to_thread(hardware.load_model_config, s.model_id, s.hf_token)
            self.model_info = hardware.model_info_from_config(s.model_id, mcfg)
        except Exception as e:
            raise RuntimeError(T('학생 모델 정보를 가져올 수 없습니다({model_id}). 모델 ID, 인터넷, HF 토큰을 확인하세요: {e}', model_id=s.model_id, e=e))
        hw = await asyncio.to_thread(hardware.detect, self.project.dir)
        avg = int(spec.get("avg_answer_tokens", 250)) + 100
        seq = s.max_seq_len or min(4096, max(256, 1 << (int(avg * 1.8) + 63).bit_length()))
        batch = s.batch_size or 1  # 가능 여부는 가장 가벼운 배치 기준
        rank = s.rank or 16
        n_total = sum(c["count"] for c in spec["categories"])
        reserved = await self._local_llm_memory_gb()

        engine = s.engine
        if engine == "auto":
            probe = hardware.check(hw, self.model_info, "mlx", s.mode, batch, seq, rank, reserved, n_total * avg)
            feas = {"mlx": probe.feasible_modes, **probe.other_engine_feasible}
            engine = await pick_auto_engine(s.mode, feas)
            self.log(T('학습 엔진 자동 선택: {engine}', engine=engine))
        eng = get_engine(engine)
        if not eng.is_installed():
            raise RuntimeError(T("선택한 학습 엔진 '{name}'이 설치되어 있지 않습니다. 설정 탭에서 설치하세요.", name=eng.name))
        supported = await eng.supported_modes()
        report = hardware.check(hw, self.model_info, engine, s.mode, batch, seq, rank, reserved, n_total * avg)
        report.feasible_modes = [m for m in report.feasible_modes if m in supported]
        if s.mode != "auto" and s.mode not in supported:
            report.messages.append(T("설치된 {name} 버전은 '{mode}' 모드를 지원하지 않습니다.", name=eng.name, mode=s.mode))
            report.ok = False
        elif s.mode == "auto" and not report.feasible_modes:
            report.ok = False
        elif s.mode != "auto":
            report.ok = report.ok and s.mode in report.feasible_modes
        self.engine_id = engine
        self.report = report
        write_json(self.project.path(F_HW), report.to_dict())
        self.log(hardware.format_report_table(report))
        if not report.ok:
            self.rep.stage("hwcheck", 1, 1, T('사양 부족 — 모든 작업 취소'))
            self.log(T('컴퓨터 사양이 학생 모델 기동·학습에 부족하여 모든 작업을 취소합니다.'), "ERROR")
            self.cancel.cancel(T('컴퓨터 사양 부족'))
            raise SpecInsufficientError(report)
        self.rep.stage("hwcheck", 1, 1, T('통과 — 가능 모드: {join}', join=', '.join(report.feasible_modes)))

    async def _local_llm_memory_gb(self) -> float:
        """데이터 생성 중 같은 Mac에서 돌아가는 로컬 교사/오케스트레이터 모델의 메모리(Ollama만 측정 가능)."""
        total = 0
        seen = set()
        for r in ROLES:
            rc = self.cfg.roles[r]
            if is_ollama_url(rc.base_url) and rc.model not in seen:
                seen.add(rc.model)
                try:
                    models = await ollama_models(ollama_root(rc.base_url))
                    size = models.get(rc.model) or models.get(f"{rc.model}:latest") or 0
                    total += size * 1.2
                except Exception:
                    pass
        return total / hardware.GB

    def _baseline_parallel_ok(self) -> bool:
        r = self.report
        if not (self.cfg.student.baseline_parallel and r):
            return False
        return r.inference_gb + r.reserved_for_local_llm_gb <= r.available_gb

    # ------------------------------------------------------------------ 학생 모델 준비 (병렬)
    async def stage_download(self) -> Path:
        s = self.cfg.student
        self.rep.stage("download", 0, 1, T('다운로드 시작'))

        def prog(done: int, total: int) -> None:
            if total:
                self.rep.stage("download", done, total, f"{done / 1e9:.2f} / {total / 1e9:.2f} GB")

        path = await download_model(s.model_id, s.hf_token, self.rep, self.cancel, prog,
                                    activity=pipeline_activity("download"))
        self.rep.stage("download", 1, 1, T('완료'))
        self.log(T('학생 모델 준비 완료: {path}', path=path))
        return path

    # ------------------------------------------------------------------ 3~5. 교사1 · 교사2 · 평가 (병렬 스트리밍)
    async def stage_generate(self, plan: dict, spec: dict) -> None:
        pc = self.cfg.pipeline
        P = self.project.path
        targets = {c["name"]: c["count"] for c in spec["categories"]}
        cats = {c["name"]: c for c in spec["categories"]}
        total_target = sum(targets.values())

        # ---- 이전 실행 상태 복원(체크포인트)
        items: dict[str, dict] = {}
        for row in read_jsonl(P(F_T1)):
            items[row["id"]] = {**items.get(row["id"], {}), **row}
        for row in read_jsonl(P(F_T2)):
            if row["id"] in items:
                items[row["id"]]["answer_2"] = row["answer_2"]
                items[row["id"]]["attempt_2"] = row.get("attempt", 0)
        final: dict[str, dict] = {}
        for row in read_jsonl(P(F_JUDGE_LOG)):
            if row.get("status") in TERMINAL:
                final[row["id"]] = row
        dedup = Deduper(pc.dedup_threshold)
        for it in items.values():
            dedup.add(it["question"])
        self.accepted = [{**items[i], "answer": f["answer"]} for i, f in final.items() if f["status"] == "accepted" and i in items]
        next_id = len(items)

        q2: asyncio.Queue = asyncio.Queue()
        q3: asyncio.Queue = asyncio.Queue()
        for it in items.values():
            if it["id"] in final:
                continue
            if it.get("answer_2") and it.get("attempt_2", 0) == it.get("attempt", 0):
                q3.put_nowait(it["id"])
            else:
                q2.put_nowait(it["id"])

        stats = {"produced": len(items), "answered": sum(1 for i in items.values() if i.get("answer_2")),
                 "judged": len(final), "target": total_target}
        g1, g2, g3 = _ErrorGuard(T('교사1')), _ErrorGuard(T('교사2')), _ErrorGuard(T('오케스트레이터'))

        def report() -> None:
            acc = len(self.accepted)
            self.rep.stage("teacher1", min(stats["produced"], stats["target"]), stats["target"],
                           T('생성 {produced} / 목표 {target}', produced=stats['produced'], target=stats['target']))
            self.rep.stage("teacher2", stats["answered"], max(1, stats["produced"]), T('답변 {answered}', answered=stats['answered']))
            rate = acc / max(1, stats["judged"]) * 100
            self.rep.stage("judge", stats["judged"], max(1, stats["produced"]),
                           T('평가 {judged} · 채택 {acc} (통과율 {rate:.0f}%)', judged=stats['judged'], acc=acc, rate=rate))

        def pending(cat: str) -> int:
            return sum(1 for i in items.values() if i["category"] == cat and i["id"] not in final)

        def accepted_in(cat: str) -> int:
            return sum(1 for a in self.accepted if a["category"] == cat)

        # ---- 교사1: 질문 + 답변1
        async def produce(cat: str, need: int) -> None:
            nonlocal next_id
            made, empty_rounds = 0, 0
            while made < need and empty_rounds < 3:
                await self.cancel.checkpoint()
                n = min(pc.teacher1_batch, need - made)
                avoid = [i["question"] for i in items.values() if i["category"] == cat]
                try:
                    gen = await self.t1.generate(plan, spec, cats[cat], n, avoid)
                    g1.ok()
                except Cancelled:
                    raise
                except (LLMError, ValueError) as e:
                    self.log(T('교사1 생성 실패({cat}): {e}', cat=cat, e=e), "WARN")
                    g1.fail(e)
                    empty_rounds += 1
                    continue
                new = 0
                for g in gen:
                    if made >= need:
                        break
                    if not length_ok(g["question"], g["answer"]) or not dedup.add(g["question"]):
                        continue
                    iid = f"q{next_id:05d}"
                    next_id += 1
                    it = {"id": iid, "category": cat, "difficulty": g["difficulty"], "question": g["question"],
                          "answer_1": g["answer"], "attempt": 0}
                    items[iid] = it
                    append_jsonl(P(F_T1), it)
                    await q2.put(iid)
                    made += 1
                    new += 1
                    stats["produced"] += 1
                empty_rounds = empty_rounds + 1 if new == 0 else 0
                report()
            if made < need:
                self.log(T("'{cat}' 카테고리에서 새로운 문항을 더 만들지 못했습니다({made}/{need}).", cat=cat, made=made, need=need), "WARN")

        # ---- 교사2: 질문만 보고 독립 답변2
        async def teacher2_worker() -> None:
            while True:
                iid = await q2.get()
                try:
                    await self.cancel.checkpoint()
                    it = items[iid]
                    try:
                        a2 = await self.t2.answer(plan, spec, it["question"])  # 답변1은 전달하지 않음
                        g2.ok()
                    except Cancelled:
                        raise
                    except (LLMError, ValueError) as e:
                        self.log(T('교사2 답변 실패({iid}): {e}', iid=iid, e=e), "WARN")
                        g2.fail(e)
                        await finish(iid, {"status": "error", "reason": T('교사2 오류: {e}', e=e)})
                        continue
                    it["answer_2"] = a2
                    it["attempt_2"] = it.get("attempt", 0)
                    append_jsonl(P(F_T2), {"id": iid, "question": it["question"], "answer_2": a2,
                                           "attempt": it["attempt_2"]})
                    stats["answered"] += 1
                    report()
                    await q3.put(iid)
                finally:
                    q2.task_done()

        async def finish(iid: str, rec: dict) -> None:
            it = items[iid]
            rec = {"id": iid, "category": it["category"], "attempt": it.get("attempt", 0), **rec}
            append_jsonl(P(F_JUDGE_LOG), rec)
            final[iid] = rec
            stats["judged"] += 1
            if rec["status"] == "accepted":
                self.accepted.append({**it, "answer": rec["answer"]})
            report()

        # ---- 오케스트레이터: 블라인드 비교 평가
        async def judge_worker() -> None:
            while True:
                iid = await q3.get()
                try:
                    await self.cancel.checkpoint()
                    it = items[iid]
                    can_regen = it.get("attempt", 0) < pc.max_regenerations
                    try:
                        v = await judge_pair(self.orch, plan, spec, it, it["answer_1"], it["answer_2"],
                                             pc.pass_threshold, can_regen, self.rng)
                        g3.ok()
                    except Cancelled:
                        raise
                    except (LLMError, ValueError) as e:
                        self.log(T('평가 실패({iid}): {e}', iid=iid, e=e), "WARN")
                        g3.fail(e)
                        await finish(iid, {"status": "error", "reason": T('평가 오류: {e}', e=e)})
                        continue
                    base = {"status": v.status, "score_1": round(v.score_1, 2), "score_2": round(v.score_2, 2),
                            "question_score": round(v.question_score, 2), "contradiction": v.contradiction,
                            "reason": v.reason, "difficulty": it.get("difficulty"), "raw": v.raw}
                    if v.status == "regenerate":
                        append_jsonl(P(F_JUDGE_LOG), {"id": iid, "attempt": it.get("attempt", 0), **base})
                        await self.cancel.checkpoint()
                        it["attempt"] = it.get("attempt", 0) + 1
                        try:
                            it["answer_1"], it["answer_2"] = await asyncio.gather(
                                self.t1.answer(plan, spec, it["question"]), self.t2.answer(plan, spec, it["question"]))
                        except Cancelled:
                            raise
                        except (LLMError, ValueError) as e:
                            await finish(iid, {"status": "error", "reason": T('재생성 오류: {e}', e=e)})
                            continue
                        it["attempt_2"] = it["attempt"]
                        append_jsonl(P(F_T1), {k: it[k] for k in ("id", "category", "difficulty", "question",
                                                                     "answer_1", "attempt")})
                        append_jsonl(P(F_T2), {"id": iid, "question": it["question"], "answer_2": it["answer_2"],
                                               "attempt": it["attempt"]})
                        self.log(T('{iid}: {reason} → 재생성 ({attempt}회차)', iid=iid, reason=v.reason, attempt=it['attempt']))
                        await q3.put(iid)
                        continue
                    if v.status == "accepted":
                        chosen = it["answer_1"] if v.chosen == "1" else it["answer_2"]
                        other = it["answer_2"] if v.chosen == "1" else it["answer_1"]
                        if v.clear_winner:
                            append_jsonl(P(F_PREF), {"id": iid, "system": spec.get("system_prompt", ""),
                                                     "prompt": it["question"], "chosen": chosen, "rejected": other})
                        await finish(iid, {**base, "chosen": v.chosen, "answer": chosen})
                    else:
                        await finish(iid, base)
                finally:
                    q3.task_done()

        n2 = self._concurrency("teacher2")
        n3 = self._concurrency("orchestrator")
        workers = [asyncio.create_task(teacher2_worker()) for _ in range(n2)] + \
                  [asyncio.create_task(judge_worker()) for _ in range(n3)]
        self.log(T('생성 시작: 교사2 작업자 {n2}개, 평가 작업자 {n3}개가 병렬로 동작합니다.', n2=n2, n3=n3))
        report()

        async def wait_queues() -> None:
            """큐가 비워질 때까지 기다리되, 작업자에서 오류가 나면 즉시 전파."""
            joiner = asyncio.create_task(_join_all(q2, q3))
            done, _ = await asyncio.wait([joiner, *workers], return_when=asyncio.FIRST_COMPLETED)
            if joiner not in done:
                joiner.cancel()
                for w in done:
                    w.result()  # 예외 전파
                raise RuntimeError(T('작업자가 예기치 않게 종료되었습니다.'))

        try:
            for rnd in range(pc.max_topup_rounds + 1):
                needs = {c: targets[c] - accepted_in(c) - pending(c) for c in targets}
                needs = {c: n for c, n in needs.items() if n > 0}
                if rnd > 0:
                    if not needs:
                        break
                    self.log(T('추가 생성 {rnd}라운드: ', rnd=rnd) + ", ".join(T('{c} {n}개', c=c, n=n) for c, n in needs.items()))
                    stats["target"] = stats["produced"] + sum(needs.values())
                producers = [asyncio.create_task(produce(c, n)) for c, n in needs.items()]
                if producers:
                    prod_all = asyncio.ensure_future(asyncio.gather(*producers))
                    done, _ = await asyncio.wait([prod_all, *workers], return_when=asyncio.FIRST_COMPLETED)
                    if prod_all not in done:
                        prod_all.cancel()
                        for w in done:
                            w.result()  # 작업자 예외 전파
                    prod_all.result()
                await wait_queues()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        self._write_quality_report(spec, items, final)
        acc = len(self.accepted)
        ratio = acc / max(1, total_target)
        self.log(T('생성·평가 완료: 채택 {acc} / 목표 {total_target} ({v:.0f}%)', acc=acc, total_target=total_target, v=ratio * 100))
        if acc < 4:
            raise RuntimeError(T('채택된 학습 데이터가 너무 적습니다(4개 미만). 교사/오케스트레이터 모델이나 기준 점수를 조정하세요.'))
        if ratio < pc.min_pass_ratio:
            msg = (T('채택된 데이터가 목표의 {v:.0f}%({acc}/{total_target})로 기준({v2:.0f}%)보다 적습니다. 이대로 학습을 계속할까요?', v=ratio * 100, acc=acc, total_target=total_target, v2=pc.min_pass_ratio * 100))
            if not self.ask_user or not await self.ask_user(T('데이터 부족'), msg):
                raise Cancelled(T('데이터 부족으로 사용자가 중단'))

    def _write_quality_report(self, spec: dict, items: dict, final: dict) -> None:
        by_status = Counter(f["status"] for f in final.values())
        per_cat = defaultdict(lambda: {"target": 0, "accepted": 0, "difficulty": Counter()})
        for c in spec["categories"]:
            per_cat[c["name"]]["target"] = c["count"]
        for a in self.accepted:
            per_cat[a["category"]]["accepted"] += 1
            per_cat[a["category"]]["difficulty"][a.get("difficulty", "medium")] += 1
        acc_final = [f for f in final.values() if f["status"] == "accepted"]
        chosen = Counter(f.get("chosen") for f in acc_final)
        rep = {
            "total_questions": len(items),
            "judged": len(final),
            "status_counts": dict(by_status),
            "accepted": len(self.accepted),
            "pass_rate": round(len(self.accepted) / max(1, len(final)), 3),
            "chosen_teacher1": chosen.get("1", 0),
            "chosen_teacher2": chosen.get("2", 0),
            "avg_score_teacher1": round(sum(f.get("score_1", 0) for f in acc_final) / max(1, len(acc_final)), 2),
            "avg_score_teacher2": round(sum(f.get("score_2", 0) for f in acc_final) / max(1, len(acc_final)), 2),
            "contradictions": sum(1 for f in final.values() if f.get("contradiction")),
            "preference_pairs": len(read_jsonl(self.project.path(F_PREF))),
            "per_category": {k: {"target": v["target"], "accepted": v["accepted"], "difficulty": dict(v["difficulty"])}
                             for k, v in per_cat.items()},
        }
        write_json(self.project.path(F_QUALITY), rep)

    # ------------------------------------------------------------------ 학습 전 기준 응답
    def _eval_items(self) -> list[dict]:
        k = self.cfg.pipeline.eval_holdout
        k = min(k, max(1, len(self.accepted) // 10)) if self.gen_done.is_set() else k
        return self.accepted[:k]

    async def _student_answers(self, questions: list[str], system: str, adapter: Path | None, max_tokens: int) -> list[str]:
        eng = get_engine(self.engine_id)
        model_path = self.model_path or await self.download_task
        server = StudentServer(lambda port: eng.serve_command(model_path, adapter, port), self.rep, self.cancel,
                               model_name=eng.serve_model_name, activity=pipeline_activity("student"))
        out: list[str] = []
        async with server:
            prov = OpenAICompatibleProvider(ProviderConfig(kind="openai", base_url=server.base_url,
                                                           model=server.model_name, temperature=0.0,
                                                           max_tokens=max_tokens, concurrency=1, timeout=900),
                                            cancel=self.cancel, activity=pipeline_activity("student"))
            try:
                for q in questions:
                    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
                    try:
                        out.append(await prov.chat(msgs))
                    except LLMError as e:
                        out.append(T('(생성 실패: {e})', e=e))
            finally:
                await prov.aclose()
        return out

    async def stage_baseline(self, spec: dict, wait_for_data: bool) -> None:
        state = self.project.state()
        if state.get("baseline"):
            self.rep.stage("baseline", 1, 1, T('기존 기준 응답 사용'))
            return
        self.model_path = self.model_path or await self.download_task
        if wait_for_data:
            while len(self.accepted) < self.cfg.pipeline.eval_holdout and not self.gen_done.is_set():
                await asyncio.sleep(2.0)
                self.cancel.check()
        items = self._eval_items()
        if not items:
            return
        self.rep.stage("baseline", 0, len(items), T('학생 모델(학습 전) 응답 생성'))
        self.log(T('학습 전 기준 응답 생성: {len}문항', len=len(items)) + (T(' (데이터 생성과 병렬)') if wait_for_data else ""))
        max_tok = min(1024, int(spec.get("avg_answer_tokens", 250) * 2))
        answers = await self._student_answers([i["question"] for i in items], spec.get("system_prompt", ""), None, max_tok)
        self.project.update_state(eval_ids=[i["id"] for i in items],
                                  baseline={i["id"]: a for i, a in zip(items, answers)})
        self.rep.stage("baseline", len(items), len(items), T('완료'))

    # ------------------------------------------------------------------ 6~8. 학습 + 평가
    def _fits(self, mode: str, batch: int, seq: int, rank: int) -> bool:
        assert self.model_info and self.report
        est = hardware.estimate_training_memory(self.model_info, self.engine_id, mode, batch, seq, rank)
        return est.total_gb <= self.report.available_gb

    def _build_dataset(self, spec: dict, cfg: algorithm.TrainConfig) -> dict:
        system = spec.get("system_prompt", "")
        eval_ids = set(self.project.state().get("eval_ids", []))
        rows = []
        dropped = 0
        for a in self.accepted:
            toks = approx_tokens(system + a["question"] + a["answer"]) + 32
            if toks > cfg.max_seq_len:
                dropped += 1
                continue
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": a["question"]}, {"role": "assistant", "content": a["answer"]}]
            rows.append((a["id"], {"messages": msgs}))
        self.rng.shuffle(rows)
        valid = [r for i, r in rows if i in eval_ids]
        train = [r for i, r in rows if i not in eval_ids]
        want_valid = max(cfg.batch_size, math.ceil(len(rows) * 0.1))
        while len(valid) < want_valid and len(train) > cfg.batch_size:
            valid.append(train.pop())
        if len(train) < cfg.batch_size or len(valid) < 1:
            raise RuntimeError(T('학습/검증 데이터가 배치 크기보다 적습니다. 데이터 수를 늘리세요.'))
        cfg.batch_size = max(1, min(cfg.batch_size, len(valid), len(train)))
        write_jsonl(self.project.path(F_TRAIN), train)
        write_jsonl(self.project.path(F_VALID), valid)
        dd = self.project.data_dir()
        dd.mkdir(exist_ok=True)
        write_jsonl(dd / "train.jsonl", train)
        write_jsonl(dd / "valid.jsonl", valid)
        if cfg.mode in ("dpo", "orpo"):
            pairs = read_jsonl(self.project.path(F_PREF))
            pairs = [{k: p[k] for k in ("system", "prompt", "chosen", "rejected")} for p in pairs]
            self.rng.shuffle(pairs)
            nv = max(cfg.batch_size, len(pairs) // 10)
            if len(pairs) - nv < cfg.batch_size:
                raise RuntimeError(T('선호도 학습에 필요한 쌍이 부족합니다({len}개).', len=len(pairs)))
            pd = self.project.sub(D_DATA) / "train_data_pref"
            pd.mkdir(exist_ok=True)
            write_jsonl(pd / "train.jsonl", pairs[nv:])
            write_jsonl(pd / "valid.jsonl", pairs[:nv])
            train, valid = pairs[nv:], pairs[:nv]  # 스텝 수는 선호도 쌍 기준
        cfg.n_train, cfg.n_valid = len(train), len(valid)
        cfg.iters = max(1, math.ceil(cfg.epochs * len(train) / cfg.batch_size))
        cfg.steps_per_eval = max(5, cfg.iters // 5)
        cfg.steps_per_report = max(1, min(10, cfg.iters // 20 or 1))
        return {T('채택 문항'): len(self.accepted), T('학습'): len(train), T('검증(학습 전후 비교 포함)'): len(valid),
                T('길이 초과로 제외'): dropped,
                T('평균 토큰(근사)'): int(sum(approx_tokens(a["question"] + a["answer"]) for a in self.accepted)
                                 / max(1, len(self.accepted)))}

    async def stage_train_and_eval(self, plan: dict, spec: dict) -> None:
        assert self.model_info and self.report and self.model_path
        eng = get_engine(self.engine_id)
        eng.activity = pipeline_activity("train")
        s = self.cfg.student
        avg = int(sum(approx_tokens(a["question"] + a["answer"]) for a in self.accepted) / max(1, len(self.accepted)))
        overrides = {"mode": s.mode, "rank": s.rank, "learning_rate": s.learning_rate, "epochs": s.epochs,
                     "batch_size": s.batch_size, "max_seq_len": s.max_seq_len, "lr_sweep": s.lr_sweep}
        rec = algorithm.recommend(plan.get("goal_type", "adaptation"), len(self.accepted), avg + 40, self.model_info,
                                  self.report, self.engine_id, overrides, self._fits)
        cfg = rec.config
        if cfg.mode not in self.report.feasible_modes:
            raise RuntimeError(T('추천된 모드 {mode}가 사양 검사에서 가능한 모드({feasible_modes})에 없습니다.', mode=cfg.mode, feasible_modes=self.report.feasible_modes))
        stats = self._build_dataset(spec, cfg)
        data_dir = self.project.sub(D_DATA) / ("train_data_pref" if cfg.mode in ("dpo", "orpo") else "train_data")
        train_rep = FileLogReporter(self.rep, self.project.path(F_TRAIN_LOG))
        model_path = await eng.prepare_model(self.model_path, cfg, self.project.sub(D_TRAIN), train_rep, self.cancel)

        # ---- 학습률 탐색
        sweep_results: list[dict] = []
        sweep_iters = min(80, max(20, cfg.iters // 3))
        if cfg.lr_sweep and cfg.iters >= 30:
            cands = algorithm.lr_candidates(cfg.learning_rate)
            self.log(T('학습률 탐색: {join} (각 {sweep_iters} step)', join=', '.join(f'{x:.1e}' for x in cands), sweep_iters=sweep_iters))
            for k, lr in enumerate(cands):
                await self.cancel.checkpoint()
                sc = dataclasses.replace(cfg, learning_rate=lr, iters=sweep_iters, steps_per_eval=sweep_iters,
                                         steps_per_report=max(1, sweep_iters // 4))
                out = self.project.sub(D_TRAIN) / "lr_sweep" / f"lr_{lr:.1e}"

                def on_p(p: TrainProgress, k=k) -> None:
                    self.rep.stage("lr_sweep", k * sweep_iters + p.step, len(cands) * sweep_iters,
                                   f"LR {lr:.1e}: step {p.step}/{sweep_iters}")

                try:
                    r = await eng.train(sc, model_path, data_dir, out, train_rep, self.cancel, on_p)
                    vl = r.final_val_loss
                except Cancelled:
                    raise
                except Exception as e:
                    self.log(T('LR {lr:.1e} 탐색 실패: {e}', lr=lr, e=e), "WARN")
                    vl = None
                sweep_results.append({"lr": lr, "val_loss": vl})
            shutil.rmtree(self.project.sub(D_TRAIN) / "lr_sweep", ignore_errors=True)
            ok = [x for x in sweep_results if x["val_loss"] is not None and math.isfinite(x["val_loss"])]
            if ok:
                best = min(ok, key=lambda x: x["val_loss"])
                cfg.learning_rate = best["lr"]
                rec.rationale.append(T('학습률 탐색 결과 검증 손실이 가장 낮은 {lr:.1e} 선택', lr=best['lr']))
            self.rep.stage("lr_sweep", 1, 1, T('선택 LR {learning_rate:.1e}', learning_rate=cfg.learning_rate))
        else:
            self.rep.stage("lr_sweep", 1, 1, T('생략'))

        # ---- 03_algorithm.md 확정본
        draft = self.project.path(F_ALGO_DRAFT).read_text(encoding="utf-8") if self.project.exists(F_ALGO_DRAFT) else ""
        final_sec = algorithm.render_markdown(T('확정본 — 품질 평가 후 실제 데이터 기준'), plan, rec, self.report,
                                              sweep_results, stats)
        self.project.path(F_ALGO).write_text(algorithm.algorithm_document(plan, draft, final_sec), encoding="utf-8")

        # ---- 본 학습
        out_dir = self.project.adapters_dir()
        shutil.rmtree(out_dir, ignore_errors=True)

        def on_train(p: TrainProgress) -> None:
            info = f"step {p.step}/{cfg.iters}"
            if p.train_loss is not None:
                info += T(' · 학습 손실 {train_loss:.3f}', train_loss=p.train_loss)
            if p.val_loss is not None:
                info += T(' · 검증 손실 {val_loss:.3f}', val_loss=p.val_loss)
            if p.peak_mem_gb:
                info += T(' · 최대 메모리 {peak_mem_gb:.1f}GB', peak_mem_gb=p.peak_mem_gb)
            self.rep.stage("train", min(p.step, cfg.iters), cfg.iters, info)

        self.rep.stage("train", 0, cfg.iters, T('학습 시작'))
        result = await eng.train(cfg, model_path, data_dir, out_dir, train_rep, self.cancel, on_train)
        self.rep.stage("train", cfg.iters, cfg.iters, T('완료 · 검증 손실 {final_val_loss}', final_val_loss=result.final_val_loss))
        if result.peak_mem_gb:
            hw = read_json(self.project.path(F_HW), {})
            hw["measured_peak_gb"] = result.peak_mem_gb
            write_json(self.project.path(F_HW), hw)
            self.log(T('실제 최대 메모리 {peak_mem_gb:.2f}GB (추정치와 비교해 계수 보정에 활용)', peak_mem_gb=result.peak_mem_gb))
        self.project.update_state(train={"engine": self.engine_id, "model_path": str(model_path),
                                         "adapter_path": str(out_dir), "config": cfg.to_dict(),
                                         "final_val_loss": result.final_val_loss,
                                         "final_train_loss": result.final_train_loss})

        # ---- 학습 후 평가
        await self.stage_eval(spec, out_dir)
        write_provenance(self.project.path(F_PROV), self.cfg, plan, stats, cfg.to_dict(), self.engine_id)

    async def stage_eval(self, spec: dict, adapter: Path) -> None:
        st = self.project.state()
        ids = st.get("eval_ids", [])
        base = st.get("baseline", {})
        by_id = {a["id"]: a for a in self.accepted}
        items = [by_id[i] for i in ids if i in by_id]
        if not items:
            self.rep.stage("eval", 1, 1, T('평가 문항 없음'))
            return
        self.rep.stage("eval", 0, len(items), T('학습 후 응답 생성'))
        max_tok = min(1024, int(spec.get("avg_answer_tokens", 250) * 2))
        after = await self._student_answers([i["question"] for i in items], spec.get("system_prompt", ""), adapter, max_tok)
        results = []
        for k, (it, aft) in enumerate(zip(items, after)):
            await self.cancel.checkpoint()
            bef = base.get(it["id"], "")
            swap = self.rng.random() < 0.5
            x, y = (aft, bef) if swap else (bef, aft)
            try:
                r = await self.orch.compare_before_after(it["question"], it["answer"], x, y)
                sx, sy = float(r.get("X", 0)), float(r.get("Y", 0))
                b_score, a_score = (sy, sx) if swap else (sx, sy)
                reason = r.get("reason", "")
            except (LLMError, ValueError, TypeError) as e:
                b_score = a_score = None
                reason = T('평가 실패: {e}', e=e)
            results.append({"id": it["id"], "question": it["question"], "reference": it["answer"], "before": bef,
                            "after": aft, "before_score": b_score, "after_score": a_score, "reason": reason})
            self.rep.stage("eval", k + 1, len(items), T('{v}/{len} 비교', v=k + 1, len=len(items)))
        scored = [r for r in results if r["before_score"] is not None]
        summary = {"avg_before": round(sum(r["before_score"] for r in scored) / max(1, len(scored)), 2),
                   "avg_after": round(sum(r["after_score"] for r in scored) / max(1, len(scored)), 2),
                   "n": len(results), "items": results}
        write_json(self.project.path(F_EVAL), summary)
        self.log(T('학습 전후 비교: 평균 {avg_before} → {avg_after} (10점 만점, {len}문항)', avg_before=summary['avg_before'], avg_after=summary['avg_after'], len=len(scored)))

    async def _unload_local_models(self) -> None:
        for r in ROLES:
            rc = self.cfg.roles[r]
            if is_ollama_url(rc.base_url):
                await ollama_unload(rc.model, ollama_root(rc.base_url))
        self.log(T('학습 전에 로컬 교사/오케스트레이터 모델을 메모리에서 내렸습니다.'))


async def _join_all(*queues: asyncio.Queue) -> None:
    # 교사2는 q3에 넣은 뒤 task_done, 재생성은 q3에 다시 넣은 뒤 task_done 하므로 순서대로 join 하면 충분하다.
    for q in queues:
        await q.join()
