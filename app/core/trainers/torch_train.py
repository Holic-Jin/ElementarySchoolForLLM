"""PyTorch + PEFT 학습 스크립트 (서브프로세스로 실행). Apple GPU(MPS) 사용.

사용: python torch_train.py <config.json>
출력 형식은 mlx-lm과 같게 맞춰 진행률 파서를 공유한다:
  Iter 10: Train loss 2.345, Learning Rate 1.000e-04, Tokens/sec 300.1, Peak mem 3.456 GB
  Iter 20: Val loss 2.100
"""
import json
import math
import random
import sys
import time
from pathlib import Path


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
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.get("seed", 42))
    random.seed(cfg.get("seed", 42))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **_dtype_kw(dtype))
    mode = cfg["mode"]
    if mode != "full":
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg.get("dropout", 0.0),
                          target_modules="all-linear", task_type="CAUSAL_LM",
                          use_dora=(mode == "dora"), use_rslora=(mode == "rslora"),
                          init_lora_weights=("pissa" if mode == "pissa" else True))
        model = get_peft_model(model, lcfg)
        if mode == "pissa":
            # 학습 후 원본 베이스에 바로 적용 가능한 LoRA로 변환하기 위해 초기 어댑터를 저장
            init_dir = Path(cfg["out_dir"]) / "pissa_init"
            model.peft_config["default"].init_lora_weights = True
            model.save_pretrained(str(init_dir))
            model.peft_config["default"].init_lora_weights = "pissa"
        # 학습 파라미터는 fp32로 유지
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    if cfg.get("grad_checkpoint", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)

    def encode(row):
        msgs = row["messages"]
        prompt_ids = _chat_ids(tok, msgs[:-1], True)
        full_ids = _chat_ids(tok, msgs, False)
        full_ids = full_ids[: cfg["max_seq_len"]]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        return full_ids, labels[: len(full_ids)]

    def load(name):
        rows = [json.loads(l) for l in (Path(cfg["data"]) / name).read_text(encoding="utf-8").splitlines() if l.strip()]
        return [encode(r) for r in rows]

    train, valid = load("train.jsonl"), load("valid.jsonl")
    bs = cfg["batch_size"]

    def batches(data, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [data[j] for j in idx[i : i + bs]]
            n = max(len(x[0]) for x in chunk)
            ids = torch.full((len(chunk), n), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(chunk), n), -100, dtype=torch.long)
            att = torch.zeros((len(chunk), n), dtype=torch.long)
            for k, (a, b) in enumerate(chunk):
                ids[k, : len(a)] = torch.tensor(a)
                lab[k, : len(b)] = torch.tensor(b)
                att[k, : len(a)] = 1
            yield ids.to(device), lab.to(device), att.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.0)
    iters, report, every = cfg["iters"], cfg["steps_per_report"], cfg["steps_per_eval"]

    def evaluate():
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for ids, lab, att in batches(valid, False):
                out = model(input_ids=ids, attention_mask=att, labels=lab)
                tot += float(out.loss)
                cnt += 1
        model.train()
        return tot / max(1, cnt)

    def peak():
        try:
            return torch.mps.driver_allocated_memory() / 1024 ** 3 if device == "mps" else 0.0
        except Exception:
            return 0.0

    model.train()
    step, t0, toks, losses = 0, time.time(), 0, []
    while step < iters:
        for ids, lab, att in batches(train, True):
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss
            if not math.isfinite(float(loss)):
                print("Loss diverged (NaN/Inf). Try a lower learning rate.", flush=True)
                sys.exit(2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            toks += int(att.sum())
            losses.append(float(loss))
            if step % report == 0 or step == iters:
                dt = max(1e-6, time.time() - t0)
                print(f"Iter {step}: Train loss {sum(losses) / len(losses):.3f}, Learning Rate {cfg['learning_rate']:.3e}, "
                      f"Tokens/sec {toks / dt:.1f}, Peak mem {peak():.3f} GB", flush=True)
                losses, toks, t0 = [], 0, time.time()
            if step % every == 0 or step == iters:
                print(f"Iter {step}: Val loss {evaluate():.3f}", flush=True)
            if step >= iters:
                break

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "pissa":
        # PiSSA는 베이스가 잔차 행렬로 바뀌므로, 원본 베이스에 적용 가능한 표준 LoRA로 변환해 저장
        model.save_pretrained(str(out_dir), path_initial_model_for_weight_conversion=str(out_dir / "pissa_init"))
    else:
        model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
