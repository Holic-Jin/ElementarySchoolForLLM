"""전체 파이프라인 E2E 검증 (가짜 LLM + 실제 소형 학생 모델 학습).

ESLLM_HOME=/tmp/x ESLLM_DATA=/tmp/data python -m tests.e2e_run [--model HuggingFaceTB/SmolLM2-135M-Instruct]
"""
import argparse
import asyncio
import subprocess
import sys
import time

from app.core.cancel import CancelToken
from app.core.config import AppConfig, RoleConfig
from app.core.events import PrintReporter
from app.core.pipeline import Pipeline
from app.core.planning import finalize
from app.core.project import Project


async def main(model: str, engine: str, mode: str, batch: int = 0) -> None:
    srv = subprocess.Popen([sys.executable, "-m", "tests.mock_llm_server", "--port", "18080"])
    time.sleep(1.5)
    try:
        cfg = AppConfig()
        for r, name in (("orchestrator", "mock-orchestrator"), ("teacher1", "mock-teacher1"), ("teacher2", "mock-teacher2")):
            cfg.roles[r] = RoleConfig(preset="custom", kind="openai", base_url="http://127.0.0.1:18080/v1",
                                      model=name, concurrency=4)
        cfg.student.model_id = model
        cfg.student.engine = engine
        cfg.student.mode = mode
        cfg.student.batch_size = batch
        cfg.pipeline.eval_holdout = 6
        proj = Project.create(f"e2e_{engine}_{mode}")
        await finalize(cfg, proj, [{"role": "user", "content": "덧셈을 가르쳐 줘"}])
        await Pipeline(proj, cfg, PrintReporter(), CancelToken(), None).run()
        print("E2E_DONE", proj.dir)
    finally:
        srv.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--engine", default="mlx")
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--batch", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(main(a.model, a.engine, a.mode, a.batch))
