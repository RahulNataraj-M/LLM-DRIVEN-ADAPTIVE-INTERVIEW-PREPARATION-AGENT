import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import finetuning.train_lora as _TL  # noqa: F401  (path helpers stay in train_lora)
from finetuning.dataset_builder import SYSTEM as _SYSTEM


def _default_adapter() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter")


def available(out: str | None = None) -> bool:
    out = out or _default_adapter()
    return os.path.exists(os.path.join(out, "adapter_model.safetensors")) or \
        os.path.exists(os.path.join(out, "adapter_model.bin"))


def evaluate_structured(question: str, answer: str, topic: str,
                        out: str | None = None) -> dict:
    out = out or _default_adapter()
    if not available(out):
        return {"available": False}
    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cfg = PeftConfig.from_pretrained(out)
    tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path,
                                                torch_dtype="auto")
    model = PeftModel.from_pretrained(base, out)
    model.eval()

    user = (f"Evaluate this candidate answer.\nTopic: {topic}\n"
            f"Question: {question}\nAnswer: {answer}")
    lines = [
        json.dumps({"role": "system", "content": _SYSTEM}),
        json.dumps({"role": "user", "content": user}),
    ]
    prompt = "".join(
        f"{json.loads(l)['role']}: {json.loads(l)['content']}\n" for l in lines) + "assistant: "
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=128, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    # extract JSON
    try:
        start = text.find('"correctness"')
        if start >= 0:
            cand = '{' + text[start:]
            for cut in ['\n\nuser:', '\nquestion:', '\n```', '\n[', '\nOutput:']:
                if cut in cand:
                    cand = cand.split(cut, 1)[0]
            m2 = re.search(r'"recommended_action"\s*:\s*"([A-Z_]+)"', cand)
            if m2:
                cand2 = cand[:m2.end()] + '}'
                return json.loads(cand2)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            j = m.group(0)
            j = j[:j.rfind('}') + 1]
            return json.loads(j)
        raise ValueError('no json')
    except Exception:
        return {"available": True, "raw": text.strip(), "parse_error": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_default_adapter())
    ap.add_argument("--question", default="Detect a cycle in a singly linked list.")
    ap.add_argument("--answer", required=True)
    ap.add_argument("--topic", default="DSA")
    args = ap.parse_args()

    if not available(args.out):
        print(json.dumps({"available": False}))
        return 1
    print(json.dumps(evaluate_structured(args.question, args.answer, args.topic,
                                         args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
