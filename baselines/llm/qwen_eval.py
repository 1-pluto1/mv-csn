"""Qwen3 zero-shot, few-shot, Clean-LoRA, and MV-CSN + Qwen3 Cascade evaluation."""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from mvcsn.data.splits import SpamRecord
from mvcsn.training.metrics import compute_classification_metrics

SYSTEM_PROMPT = (
    "你是一个专业的中文短信反垃圾风控系统。请判断用户给出的短信是否为垃圾/诈骗/黑产推广短信"
    "（可能包含同音字、形近字、偏旁拆字或符号插空混淆）。"
    "如果是垃圾短信请只输出1，如果是正常短信请只输出0，不要输出任何其他内容。"
)


def load_qwen_model(model_id: str, device: torch.device | str = "cuda"):
    """Load Qwen3 CausalLM in bfloat16 on GPU with left-padding tokenizer."""
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.config.use_cache = False
    model.eval()
    return model, tok


def get_binary_token_ids(tokenizer) -> Tuple[int, int]:
    """Return single-token IDs for '0' (ham) and '1' (spam)."""
    id_0 = tokenizer.encode("0", add_special_tokens=False)
    id_1 = tokenizer.encode("1", add_special_tokens=False)
    assert len(id_0) == 1 and len(id_1) == 1, f"Expected single tokens for 0/1, got {id_0}, {id_1}"
    return int(id_0[0]), int(id_1[0])


def format_spam_prompt(
    tokenizer,
    user_text: str,
    few_shot_examples: Optional[Sequence[Tuple[str, int]]] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Format a non-thinking Qwen3 chat prompt ready for next-token 0/1 scoring."""
    messages = [{"role": "system", "content": system_prompt}]
    if few_shot_examples:
        for ex_txt, ex_lbl in few_shot_examples:
            messages.append({"role": "user", "content": f"短信内容：{ex_txt}"})
            messages.append({"role": "assistant", "content": str(int(ex_lbl))})
    messages.append({"role": "user", "content": f"短信内容：{user_text}"})

    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if rendered.endswith("<think>\n") or rendered.endswith("<think>"):
        rendered = rendered + "\n</think>\n\n"
    return rendered


@torch.no_grad()
def score_binary_llm(
    model,
    tokenizer,
    user_texts: Sequence[str],
    batch_size: int = 32,
    max_length: int = 384,
    few_shot_examples: Optional[Sequence[Tuple[str, int]]] = None,
    raw_prompts: Optional[Sequence[str]] = None,
    device: torch.device | str = "cuda",
) -> Tuple[List[float], float]:
    """Single-forward-pass calibrated P(spam='1') scoring with synchronized latency."""
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    t0_id, t1_id = get_binary_token_ids(tokenizer)

    if raw_prompts is not None:
        prompts = list(raw_prompts)
    else:
        prompts = [format_spam_prompt(tokenizer, txt, few_shot_examples=few_shot_examples) for txt in user_texts]

    if not prompts:
        return [], 0.0

    warm = tokenizer(prompts[: min(2, len(prompts))], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
        _ = model(**warm, logits_to_keep=1, use_cache=False).logits[:, -1, [t0_id, t1_id]]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()

    probs: List[float] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            logits = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, [t0_id, t1_id]].float()
            p = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(p.cpu().tolist())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_ms = (time.time() - t_start) * 1000.0 / max(1, len(prompts))
    return probs, float(infer_ms)


class _PromptClsDataset(Dataset):
    def __init__(self, records: Sequence[SpamRecord], tokenizer, max_length: int = 320) -> None:
        prompts = [format_spam_prompt(tokenizer, r.raw_text) for r in records]
        enc = tokenizer(prompts, padding=False, truncation=True, max_length=max_length)
        self.items = [
            {"input_ids": enc["input_ids"][i], "label": int(records[i].label)}
            for i in range(len(records))
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.items[idx]


def _make_left_pad_collator(pad_token_id: int):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.zeros((bsz,), dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            L = len(ids)
            input_ids[i, max_len - L :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - L :] = 1
            labels[i] = int(b["label"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def finetune_qwen_clean_lora(
    model_id: str,
    tr_records: Sequence[SpamRecord],
    va_records: Sequence[SpamRecord],
    te_records: Sequence[SpamRecord],
    device: torch.device,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-4,
    save_dir: Optional[str] = None,
) -> Tuple[Dict[str, float], List[float], float, int]:
    """LoRA fine-tuning strictly on Clean training data, evaluated on RealVariant test set."""
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    t0_id, t1_id = get_binary_token_ids(tok)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    base_model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    total_params = sum(p.numel() for p in model.parameters())

    micro_batch = min(batch_size, 16)
    grad_accum = max(1, batch_size // micro_batch)
    tr_ds = _PromptClsDataset(tr_records, tok)
    collate_fn = _make_left_pad_collator(tok.pad_token_id)
    tr_loader = DataLoader(tr_ds, batch_size=micro_batch, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    total_opt_steps = max(1, epochs * ((len(tr_loader) + grad_accum - 1) // grad_accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_opt_steps)

    va_texts = [r.raw_text for r in va_records]
    va_labels = [r.label for r in va_records]

    best_f1 = -1.0
    best_lora_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tr_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1, use_cache=False).logits[:, -1, [t0_id, t1_id]].float()
                loss = F.cross_entropy(logits, labels) / grad_accum
            loss.backward()
            tot_loss += float(loss.item()) * grad_accum
            if step % grad_accum == 0 or step == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

        va_probs, _ = score_binary_llm(model, tok, va_texts, batch_size=32, device=device)
        va_m = compute_classification_metrics(va_labels, va_probs)
        print(f"  [LoRA epoch {epoch}/{epochs}] train_loss={tot_loss / max(1, len(tr_loader)):.4f} va_clean_f1={va_m['f1']:.4f}", flush=True)
        if va_m["f1"] > best_f1:
            best_f1 = va_m["f1"]
            best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    if best_lora_state is not None:
        model.load_state_dict(best_lora_state, strict=False)
    if save_dir is not None:
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    te_probs, te_ms = score_binary_llm(model, tok, [r.raw_text for r in te_records], batch_size=32, device=device)
    te_metrics = compute_classification_metrics([r.label for r in te_records], te_probs)
    del model, base_model, opt, sched, tr_loader
    gc.collect()
    torch.cuda.empty_cache()
    return te_metrics, te_probs, te_ms, int(total_params)


@torch.no_grad()
def score_binary_llm_with_logits(
    model,
    tokenizer,
    user_texts: Sequence[str],
    batch_size: int = 32,
    max_length: int = 384,
    few_shot_examples: Optional[Sequence[Tuple[str, int]]] = None,
    raw_prompts: Optional[Sequence[str]] = None,
    device: torch.device | str = "cuda",
) -> Tuple[List[float], List[float], float]:
    """Length-sorted single-forward-pass scoring returning (probs, logit_diffs, infer_ms)."""
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    t0_id, t1_id = get_binary_token_ids(tokenizer)

    if raw_prompts is not None:
        prompts = list(raw_prompts)
    else:
        prompts = [format_spam_prompt(tokenizer, txt, few_shot_examples=few_shot_examples) for txt in user_texts]

    if not prompts:
        return [], [], 0.0

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    sorted_prompts = [prompts[i] for i in order]

    warm = tokenizer(sorted_prompts[: min(2, len(sorted_prompts))], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
        _ = model(**warm, logits_to_keep=1, use_cache=False).logits[:, -1, [t0_id, t1_id]]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()

    s_probs: List[float] = []
    s_logits: List[float] = []
    for i in range(0, len(sorted_prompts), batch_size):
        chunk = sorted_prompts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            logits = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, [t0_id, t1_id]].float()
            ldiff = logits[:, 1] - logits[:, 0]
            p = torch.sigmoid(ldiff)
        s_probs.extend(p.cpu().tolist())
        s_logits.extend(ldiff.cpu().tolist())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_ms = (time.time() - t_start) * 1000.0 / max(1, len(prompts))

    probs = [0.0] * len(prompts)
    ldiffs = [0.0] * len(prompts)
    for pos, orig_i in enumerate(order):
        probs[orig_i] = s_probs[pos]
        ldiffs[orig_i] = s_logits[pos]
    return probs, ldiffs, float(infer_ms)
