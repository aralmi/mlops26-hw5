"""Стадия train: LoRA-дообучение на выходе стадии tokenize.

    python -m src.train --variant all_layers          # полный прогон варианта
    python -m src.train --variant all_layers --max-steps 4 --out /tmp/x   # smoke

Цикл обучения написан руками, а не через Trainer: так видно всё, что
обычно прячется, — где считается val loss, как копятся градиенты, что
сохраняется рядом с адаптером.

Что сохраняется в models/adapter_<variant>/: адаптер.
В metrics/train_<variant>.json — кривые train/val loss, время, пиковая память,
число обучаемых параметров, вес адаптера и отпечаток входов.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

# Потолок памяти Metal — до импорта torch. Без него mps занимает сколько дадут,
# и на ноутбуке с 16–32 ГБ система уходит в своп вместо внятной ошибки.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")   # ниже верхнего; остаётся запас системе

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.config import load_params
from src.data import LABEL_PAD_ID, batches, load_split
from src.runtime import allocated_bytes, memory_metric, resolve_device, resolve_dtype, set_seed


TRAIN_CODE = ("src/train.py", "src/data.py", "src/runtime.py", "src/config.py")
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants")


def inputs_fingerprint(params: dict) -> str:
    """Отпечаток кода обучения и секций конфига, от которых зависит адаптер.

    check.sh сверяет его с текущим: правили train.py или lr, а адаптер
    остался от прошлого прогона — проверять его бессмысленно.
    """
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    h.update(json.dumps({k: params.get(k) for k in TRAIN_PARAMS}, sort_keys=True).encode())
    for name in (params["data"]["train"], params["data"]["val"]):
        h.update(name.encode())
        h.update(hashlib.sha256(Path(name).read_bytes()).digest())
    return h.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    cfg = params["lora"]
    if not 0 <= freeze_first < n_layers:
        raise ValueError(f"freeze_first={freeze_first} вне диапазона 0..{n_layers - 1}")
    forbidden = {"embed_tokens", "lm_head"}
    targets = set(cfg["target_modules"])
    if forbidden & targets or cfg.get("modules_to_save"):
        raise ValueError("LoRA должен сохранять только низкоранговые проекции, не словарь")
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=cfg.get("modules_to_save"),
        layers_to_transform=list(range(freeze_first, n_layers)) if freeze_first else None,
        task_type="CAUSAL_LM",
    )


def chunked_eval_loss(model, batch: dict[str, torch.Tensor], chunk_tokens: int) -> torch.Tensor:
    """Тот же token-weighted loss при оценке, но без полного тензора logits."""
    base = model.get_base_model()
    hidden = base.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).last_hidden_state
    targets = batch["labels"][:, 1:]
    positions = (targets != LABEL_PAD_ID).any(dim=0).nonzero().flatten()
    if positions.numel() == 0:
        raise ValueError("в батче нет обучаемых токенов ответа")
    total = torch.zeros((), device=hidden.device)
    count = torch.zeros((), device=hidden.device)
    for part in positions.split(chunk_tokens):
        part_targets = targets.index_select(1, part)
        logits = base.lm_head(hidden.index_select(1, part)).float()
        total += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), part_targets.reshape(-1),
            ignore_index=LABEL_PAD_ID, reduction="sum",
        )
        count += (part_targets != LABEL_PAD_ID).sum()
    return total / count


def chunked_train_backward(model, batch: dict[str, torch.Tensor], chunk_tokens: int,
                           accumulation_size: int) -> torch.Tensor:
    """Обратный проход через замороженный lm_head по блокам, затем через LoRA.

    Градиент loss по hidden вычисляется отдельно для каждого блока словаря.
    После этого один backward передаёт собранный градиент в transformer.
    Это цепное правило без вложенного checkpoint для проекции на словарь.
    """
    base = model.get_base_model()
    if any(p.requires_grad for p in base.lm_head.parameters()):
        raise ValueError("блочный backward требует замороженный lm_head")
    if chunk_tokens <= 0 or accumulation_size <= 0:
        raise ValueError("размер блока и накопления должны быть положительными")
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False).last_hidden_state
    if not torch.isfinite(hidden).all().item():
        raise RuntimeError("hidden содержит нечисловые значения до проекции")
    targets = batch["labels"][:, 1:]
    positions = (targets != LABEL_PAD_ID).any(dim=0).nonzero().flatten()
    count = (targets != LABEL_PAD_ID).sum()
    if not positions.numel():
        raise ValueError("в батче нет обучаемых токенов ответа")
    grad_hidden = torch.zeros_like(hidden)
    total = torch.zeros((), device=hidden.device)
    for part in positions.split(chunk_tokens):
        h = hidden.detach().index_select(1, part).requires_grad_()
        logits = base.lm_head(h).float()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.index_select(1, part).reshape(-1),
            ignore_index=LABEL_PAD_ID, reduction="sum",
        ) / count / accumulation_size
        if not torch.isfinite(loss).item():
            raise RuntimeError("нечисловой loss в проекции на словарь")
        gradient, = torch.autograd.grad(loss, h)
        grad_hidden.index_copy_(1, part, gradient)
        total += loss.detach()
        del logits, loss, h, gradient
    if not torch.isfinite(grad_hidden).all().item():
        raise RuntimeError("градиент lm_head содержит нечисловые значения")
    hidden.backward(grad_hidden)
    return total


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int, chunk_tokens: int) -> float:
    """Средний лосс на токен по всему val-сплиту.

    Среднее по батчам нельзя: в батчах разное число токенов под маской.
    Поэтому сумма лоссов, взвешенная числом токенов, делённая на их сумму.
    """
    model.eval()
    if device.type == "mps":
        torch.mps.empty_cache()
    total, count = 0.0, 0
    for batch in batches(examples, batch_size, pad_id, shuffle=False, seed=0):
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum())
        if n == 0:
            continue
        loss = chunked_eval_loss(model, batch, chunk_tokens)
        total += loss.item() * n
        count += n
    model.train()
    if device.type == "mps":
        torch.mps.empty_cache()   # логиты оценки не должны висеть в кэше до конца обучения
    return total / max(count, 1)


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 2)


def accumulation_window(batch_index: int, batch_count: int, grad_accum: int) -> tuple[int, bool]:
    """Размер группы и её конец, включая последнюю неполную группу эпохи."""
    if grad_accum <= 0 or not 0 <= batch_index < batch_count:
        raise ValueError("некорректная группа накопления градиентов")
    group_start = (batch_index // grad_accum) * grad_accum
    size = min(grad_accum, batch_count - group_start)
    return size, batch_index + 1 == group_start + size


def select_examples(examples: list, limit: int | None, seed: int) -> tuple[list, list[int]]:
    """Фиксированная случайная выборка без изменения исходных тензоров."""
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError("размер выборки должен быть положительным целым числом или null")
    indices = list(range(len(examples)))
    if limit is not None and limit < len(examples):
        indices = sorted(random.Random(seed).sample(indices, limit))
    return [examples[i] for i in indices], indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="куда писать адаптер и метрики (smoke-тесты)")
    ap.add_argument("--val-limit", type=int, default=None, help="оценивать на первых N примерах val (smoke)")
    ap.add_argument("--checkpoint-dir", default=None, help="папка состояния для продолжения прогона")
    ap.add_argument("--fresh", action="store_true", help="начать заново, не читать локальное состояние")
    args = ap.parse_args()

    params = load_params()
    variants = {v["name"]: v for v in params["variants"]}
    if args.variant not in variants:
        raise SystemExit(f"нет варианта {args.variant!r}, есть: {sorted(variants)}")
    variant = variants[args.variant]
    tcfg = params["train"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps должен быть положительным")

    # До загрузки модели и инициализации матриц LoRA: иначе сид опаздывает.
    set_seed(tcfg["seed"])

    device = resolve_device(params["model"]["device"])
    if device.type == "cpu":
        torch.set_num_threads(params["model"].get("cpu_threads", 4))
    dtype = resolve_dtype(params["model"]["dtype"])

    train_blob = load_split(params["data"]["train"])
    val_blob = load_split(params["data"]["val"])
    source_counts = {"train": len(train_blob["examples"]), "val": len(val_blob["examples"])}
    subset_seed = tcfg.get("subset_seed", tcfg["seed"])
    train_blob["examples"], train_indices = select_examples(
        train_blob["examples"], tcfg.get("train_limit"), subset_seed)
    val_blob["examples"], val_indices = select_examples(
        val_blob["examples"], tcfg.get("val_limit"), subset_seed + 1)
    if args.val_limit:
        val_blob["examples"] = val_blob["examples"][:args.val_limit]
        val_indices = val_indices[:args.val_limit]
    pad_id = train_blob["pad_token_id"]

    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        params["model"]["name"], dtype=dtype,
        attn_implementation=params["model"].get("attn_implementation", "sdpa"),
    ).to(device)
    n_layers = model.config.num_hidden_layers
    if params["train"].get("gradient_checkpointing"):
        # Активации 28 слоёв не храним, а пересчитываем на обратном проходе:
        # памяти в разы меньше, шаг примерно на треть дольше.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, lora_config(params, n_layers, variant["freeze_first"]))
    # Non-reentrant checkpoint вычисляет градиенты параметров LoRA без
    # градиента embedding. В freeze14 нижние слои не нужны в backward.
    if hasattr(model.get_base_model(), "_require_grads_hook"):
        model.get_base_model().disable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.train()

    examples = train_blob["examples"]
    micro_per_epoch = math.ceil(len(examples) / tcfg["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    if max_steps:
        total_steps = min(total_steps, max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * tcfg["warmup_ratio"])), total_steps
    )

    eval_bs = tcfg.get("eval_batch_size", tcfg["batch_size"])
    base_val = None
    curve_train: list[list[float]] = []
    curve_val: list[list[float]] = []
    print(f"[{args.variant}] устройство {device}, обучаемых {trainable:,} из {total:,} "
          f"({trainable / total:.3%}); шагов {total_steps}")

    peak = allocated_bytes(device)
    started = time.perf_counter()
    step, micro, accum_loss, diverged = 0, 0, 0.0, False
    processed_tokens = 0
    eval_seconds = 0.0
    prior_seconds, prior_eval = 0.0, 0.0
    resume_epoch, next_batch = 0, 0
    checkpoint_dir = args.checkpoint_dir or (".cache/checkpoints" if not args.out else None)
    checkpoint_path = Path(checkpoint_dir) / f"{args.variant}.pt" if checkpoint_dir else None
    identity = {"fingerprint": inputs_fingerprint(params), "total_steps": total_steps,
                "val_examples": len(val_blob["examples"])}
    if checkpoint_path and checkpoint_path.exists() and not args.fresh:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["identity"] == identity:
            set_peft_model_state_dict(model, state["adapter"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["rng_cpu"])
            if device.type == "mps":
                torch.mps.set_rng_state(state["rng_device"])
            elif device.type == "cuda":
                torch.cuda.set_rng_state_all(state["rng_device"])
            step, micro = state["step"], state["micro"]
            processed_tokens = state["processed_tokens"]
            resume_epoch, next_batch = state["epoch"], state["next_batch"]
            base_val, curve_train, curve_val = state["base_val"], state["curve_train"], state["curve_val"]
            prior_seconds, prior_eval = state["seconds"], state["eval_seconds"]
            peak = max(peak, state["peak"])
            print(f"  продолжение с шага {step}/{total_steps}", flush=True)

    def save_checkpoint(epoch, next_batch):
        if checkpoint_path is None:
            return
        rng_device = (torch.mps.get_rng_state() if device.type == "mps" else
                      torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        state = {"identity": identity, "adapter": get_peft_model_state_dict(model),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng_cpu": torch.get_rng_state(), "rng_device": rng_device,
                 "step": step, "micro": micro, "epoch": epoch, "next_batch": next_batch,
                 "processed_tokens": processed_tokens,
                 "base_val": base_val, "curve_train": curve_train, "curve_val": curve_val,
                 "seconds": prior_seconds + time.perf_counter() - started - eval_seconds,
                 "eval_seconds": prior_eval + eval_seconds, "peak": peak}
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)

    if base_val is None:
        eval_started = time.perf_counter()
        base_val = evaluate(model, val_blob["examples"], pad_id, device, eval_bs,
                            tcfg["loss_chunk_tokens"])
        eval_seconds += time.perf_counter() - eval_started
        curve_val.append([0, round(base_val, 4)])
        peak = max(peak, allocated_bytes(device))
        print(f"  шаг 0/{total_steps}: val {base_val:.4f} (базовая модель)", flush=True)
        save_checkpoint(0, 0)
    for epoch in range(resume_epoch, tcfg["epochs"]):
        if step >= total_steps:
            break
        epoch_batches = batches(examples, tcfg["batch_size"], pad_id,
                                shuffle=True, seed=tcfg["seed"] + epoch)
        for batch_index, batch in enumerate(epoch_batches):
            if epoch == resume_epoch and batch_index < next_batch:
                continue
            group_size, group_end = accumulation_window(batch_index, micro_per_epoch, tcfg["grad_accum"])
            if tcfg.get("gradient_checkpointing"):
                base = model.get_base_model()
                use_checkpoint = batch["input_ids"].shape[1] >= tcfg.get("gradient_checkpointing_min_tokens", 0)
                if use_checkpoint and not base.is_gradient_checkpointing:
                    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                elif not use_checkpoint and base.is_gradient_checkpointing:
                    base.gradient_checkpointing_disable()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = chunked_train_backward(model, batch, tcfg["loss_chunk_tokens"], group_size)
            if not torch.isfinite(loss).item():
                diverged = True
                print(f"  микро-шаг {micro + 1}: лосс {loss.item()} — обучение разошлось")
                break
            grads = [(name, p.grad) for name, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None]
            # Один sync с ускорителем, а не отдельный .item() для сотен матриц.
            finite = torch.stack([torch.isfinite(g).all() for _, g in grads]).all().item()
            if not finite:
                bad_grads = [name for name, g in grads if not torch.isfinite(g).all().item()]
                diverged = True
                print(f"  шаг {step}, микро-шаг {micro + 1}, длина {batch['input_ids'].shape[1]}: "
                      f"нечисловые градиенты в {bad_grads[:3]}", flush=True)
                break
            accum_loss += loss.item()
            micro += 1
            processed_tokens += int(batch["attention_mask"].sum().item())
            peak = max(peak, allocated_bytes(device))
            if not group_end:
                continue
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tcfg["max_grad_norm"], error_if_nonfinite=True
                )
            except RuntimeError as exc:
                cpu_norms = [(name, torch.linalg.vector_norm(g.detach().cpu().double()).item())
                             for name, g in grads]
                print("  CPU float64 нормы:", sorted(cpu_norms, key=lambda x: x[1], reverse=True)[:3],
                      "общая", math.sqrt(sum(n * n for _, n in cpu_norms)), flush=True)
                print(f"  шаг {step}, микро-шаг {micro}, длина {batch['input_ids'].shape[1]}: "
                      f"нечисловая норма градиента: {exc}", flush=True)
                raise
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            curve_train.append([step, round(accum_loss, 4)])
            print(f"  train шаг {step}/{total_steps}: {accum_loss:.4f}", flush=True)
            if not math.isfinite(accum_loss):
                diverged = True
                print(f"  шаг {step}: лосс {accum_loss} — обучение разошлось, останавливаюсь")
                break
            accum_loss = 0.0
            if step % tcfg["eval_every"] == 0 or step == total_steps:
                eval_started = time.perf_counter()
                val_loss = evaluate(model, val_blob["examples"], pad_id, device, eval_bs,
                                    tcfg["loss_chunk_tokens"])
                eval_seconds += time.perf_counter() - eval_started
                curve_val.append([step, round(val_loss, 4)])
                peak = max(peak, allocated_bytes(device))
                print(f"  шаг {step}/{total_steps}: train {curve_train[-1][1]:.4f}, "
                      f"val {val_loss:.4f}", flush=True)
                save_checkpoint(epoch, batch_index + 1)
            if step >= total_steps:
                break
        if diverged or step >= total_steps:
            break
    seconds = prior_seconds + time.perf_counter() - started - eval_seconds

    if diverged:
        raise RuntimeError("Обучение разошлось: адаптер этого прогона не сохраняется")

    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    adapter_dir = out_root / f"adapter_{args.variant}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "variant": args.variant,
        "freeze_first": variant["freeze_first"],
        "model": params["model"]["name"],
        "device": device.type,
        "dtype": params["model"]["dtype"],
        "seed": tcfg["seed"],
        "lr": tcfg["lr"],
        "effective_batch": tcfg["batch_size"] * tcfg["grad_accum"],
        "steps": step,
        "train_examples": len(examples),
        "val_examples": len(val_blob["examples"]),
        "source_examples": source_counts,
        "selection": {"method": "seeded random sample without replacement",
                      "seed": subset_seed, "val_seed": subset_seed + 1,
                      "train_indices": train_indices, "val_indices": val_indices},
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share": round(trainable / total, 6),
        "base_val_loss": base_val,
        "final_val_loss": curve_val[-1][1] if curve_val else None,
        "diverged": diverged,
        "curve_train": curve_train,
        "curve_val": curve_val,
        "seconds": round(seconds, 1),
        "eval_seconds": round(prior_eval + eval_seconds, 1),
        "seconds_per_step": round(seconds / max(step, 1), 3),
        "train_tokens": processed_tokens,
        "train_tokens_per_sec": round(processed_tokens / seconds, 1) if seconds else 0,
        "peak_memory_mb": round(peak / 1048576, 1),
        "memory_metric": memory_metric(device),
        "adapter_dir": str(adapter_dir),
        "adapter_size_mb": dir_size_mb(adapter_dir),
        "inputs_fingerprint": inputs_fingerprint(params),
    }
    mdir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.variant}] {step} шагов за {seconds:.0f} с; "
          f"пик памяти {metrics['peak_memory_mb']:.0f} МБ; адаптер {metrics['adapter_size_mb']} МБ -> {adapter_dir}")


if __name__ == "__main__":
    main()
