import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as C
from finetuning.dataset_builder import OUT as DATASET


def _load_tok_and_model(tok, m):
    """Collate/vectorise nothing; populate dataset batch manually to stay light."""
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="HF base causal-instruct model (llama is gated; use a "
                         "non-gated small instruct model for a CPU smoke run)")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "adapter"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              TrainingArguments, Trainer)
    from torch.utils.data import Dataset

    if not os.path.exists(args.dataset):
        print(f"Dataset missing: {args.dataset}. Run dataset_builder first.")
        return 1

    print("Loading tokenizer + base model:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")

    # LoRA on the attention projections that exist in this architecture.
    target = []
    for n, _p in model.named_parameters():
        if n.endswith((".q_proj.weight", ".v_proj.weight", ".k_proj.weight",
                       ".o_proj.weight")):
            target.append(n.split(".")[-2])
    target = sorted(set(target)) or None
    print("LoRA target modules:", target)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                      target_modules=target, lora_dropout=0.05)
    model = get_peft_model(model, lora)

    # ---- dataset -------------------------------------------------------

    class EvalDS(Dataset):
        def __init__(self):
            self.samples = []
            self.labels = []
            with open(args.dataset, encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            # group into examples of 3 consecutive lines (system/user/assistant)
            for i in range(0, len(lines), 3):
                grp = lines[i:i + 3]
                if len(grp) != 3:
                    break
                # prompt = system+user+"assistant: " (masked, loss ignored)
                prompt = "".join(
                    f"{r['role']}: {r['content']}\n" for r in grp[:2])
                prompt += "assistant: "
                full = prompt + grp[2]["content"] + "\n"
                ids = tok(full, truncation=True,
                          max_length=args.max_len)["input_ids"]
                plen = len(tok(prompt, truncation=True,
                               max_length=args.max_len)["input_ids"])
                self.samples.append(ids)
                self.labels.append([-100] * plen + ids[plen:])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return {"input_ids": self.samples[i], "labels": self.labels[i]}

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id or 0
        inp = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        for j, b in enumerate(batch):
            inp[j, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
            lab[j, :len(b["labels"])] = torch.tensor(b["labels"])
        return {"input_ids": inp, "labels": lab}

    ds = EvalDS()
    print(f"dataset rows: {len(ds)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    fp16 = device == "cuda" and not bf16
    print(f"Device: {device}  bf16={bf16}  fp16={fp16}")

    os.makedirs(args.out, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="linear", warmup_steps=3,
        bf16=bf16, fp16=fp16, dataloader_pin_memory=(device == "cuda"),
        save_strategy="steps", save_steps=20, logging_steps=5, report_to=[], seed=C.SEED)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collate, processing_class=tok)
    trainer.train()
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Adapter saved to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
