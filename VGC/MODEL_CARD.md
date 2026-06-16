---
license: mit
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
  - pokemon
  - reinforcement-learning
  - grpo
  - openenv
  - poke-env
  - lora
datasets: []
pipeline_tag: text-generation
---

# Hardread: GRPO-Trained Pokemon Battler

LoRA adapters for [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) trained with GRPO to play competitive Pokemon Showdown battles.

## Model Description

These adapters are trained on real rollout trajectories collected from live Pokemon Showdown battles (gen4randombattle format). The training pipeline:

1. **JSON warm-up SFT** — teaches the model to output valid `{"action": "move"|"switch", "choice": "..."}` JSON
2. **GRPO training** — optimizes the policy using shaped rewards from the Hardread environment

The environment provides dense reward signals including damage dealt/taken, knockouts, healing, setup moves, type effectiveness, and penalties for illegal actions.

## Checkpoints

| Branch | Description |
|---|---|
| `grpo-qwen3-4b-run1` | First GRPO run |
| `grpo-qwen3-4b-run2` | Second run with tuned reward shaping |
| `grpo-qwen3-4b-run3` | Third run, best performing |

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_REPO = "Atharva2099/openenv-smogon-rl"
REVISION = "grpo-qwen3-4b-run3"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER_REPO, revision=REVISION)
model.eval()

# Example battle state prompt
messages = [
    {"role": "system", "content": (
        "You are a competitive Pokemon battler. "
        "Output ONLY a single JSON object.\n"
        '{"action": "move" | "switch", "choice": "Exact Name"}'
    )},
    {"role": "user", "content": "<battle state markdown here>"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=24, temperature=0.3)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training Details

- **Base model:** Qwen3-4B-Instruct-2507
- **Method:** LoRA + GRPO (Group Relative Policy Optimization)
- **Environment:** gen4randombattle via poke-env + local Pokemon Showdown
- **Reward:** Multi-component shaped reward (damage, KOs, healing, setup, type effectiveness, illegal action penalty)
- **Max steps:** 30 per battle
- **Battle opponent:** RandomPlayer (poke-env built-in)

## Links

- **Environment repo:** [github.com/Atharva2099/Hardread](https://github.com/Atharva2099/Hardread)
- **Live demo:** [HF Space](https://huggingface.co/spaces/Atharva2099/Hardread)
- **Training notebook:** `trainer.ipynb` in the environment repo
