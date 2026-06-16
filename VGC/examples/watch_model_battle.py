from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from smogon_rl.action_space import (  # noqa: E402
    build_action_instructions,
    enumerate_actions,
    extract_action_json_from_text,
    parse_llm_action,
)
from smogon_rl.config import EnvConfig  # noqa: E402
from smogon_rl.openenv_sync_env import PokemonShowdownEnv  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = (
    "You are a competitive Pokemon battler. "
    "Analyse the battle state below and choose exactly one action. "
    "You MUST output ONLY a single JSON object — no explanation, no markdown fences, just the raw JSON.\n"
    '{"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one visible Pokemon Showdown battle with a trained adapter."
    )
    parser.add_argument(
        "--repo-id",
        default="Atharva2099/openenv-smogon-rl",
        help="Hugging Face repo containing the adapter.",
    )
    parser.add_argument(
        "--revision",
        default="grpo-qwen3-4b-run2",
        help="Adapter revision/branch to load.",
    )
    parser.add_argument(
        "--battle-format",
        default="gen4randombattle",
        help="Pokemon Showdown battle format.",
    )
    parser.add_argument(
        "--showdown-dir",
        default=os.environ.get("SHOWDOWN_DIR", ""),
        help="Local pokemon-showdown directory. Used only if the server is not already running.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Max tokens to generate for an action.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="How many candidates to sample per turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Sampling top-p.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Sampling top-k.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Environment max steps per battle.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local Showdown room in the default browser.",
    )
    return parser.parse_args()


def _server_up(host: str = "127.0.0.1", port: int = 8000) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def ensure_showdown_server(showdown_dir: str) -> subprocess.Popen[str] | None:
    if _server_up():
        return None
    if not showdown_dir:
        raise RuntimeError(
            "Pokemon Showdown is not running on port 8000 and --showdown-dir was not provided."
        )

    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=showdown_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for _ in range(20):
        if _server_up():
            return proc
        time.sleep(0.5)
    raise RuntimeError("Pokemon Showdown did not start on port 8000.")


def build_prompt_messages(state_str: str, valid_actions) -> list[dict]:
    instructions = build_action_instructions(valid_actions)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{state_str}\n\n{instructions}"},
    ]


@torch.no_grad()
def generate_action_candidates(model, tokenizer, state_str: str, valid_actions, args: argparse.Namespace) -> list[str]:
    messages = build_prompt_messages(state_str, valid_actions)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_generations,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(out[input_len:], skip_special_tokens=True) for out in outputs]


def load_model(repo_id: str, revision: str):
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=token,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model,
        repo_id,
        revision=revision,
        token=token,
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer


def choose_action(model, tokenizer, state_str: str, valid_actions, args: argparse.Namespace) -> str:
    candidates = generate_action_candidates(model, tokenizer, state_str, valid_actions, args)
    for candidate in candidates:
        clean = re.sub(r"<think>.*?</think>", "", candidate, flags=re.DOTALL).strip()
        try:
            parse_llm_action(clean, valid_actions)
            return clean
        except ValueError:
            pass

        extracted = extract_action_json_from_text(candidate)
        if extracted is not None:
            try:
                parse_llm_action(extracted, valid_actions)
                return extracted
            except ValueError:
                pass

    fallback = valid_actions[0]
    return json.dumps({"action": fallback.action_type, "choice": fallback.choice})


def main() -> None:
    args = parse_args()
    showdown_proc = ensure_showdown_server(args.showdown_dir)
    try:
        model, tokenizer = load_model(args.repo_id, args.revision)
        env = PokemonShowdownEnv(
            config=EnvConfig(
                battle_format=args.battle_format,
                max_steps_per_battle=args.max_steps,
                verbose_logging=True,
                log_every_n_steps=1,
                poll_heartbeat_seconds=5.0,
            )
        )

        state_str = env.reset()
        battle = env._ensure_battle()
        room_url = f"http://127.0.0.1:8000/{battle.battle_tag}"
        print(f"Watching room: {room_url}", flush=True)
        if args.open_browser:
            webbrowser.open(room_url)

        done = False
        total_reward = 0.0
        step_idx = 0
        while not done:
            step_idx += 1
            battle = env._ensure_battle()
            valid_actions = enumerate_actions(battle)
            if not valid_actions:
                print("No valid actions available; ending battle.")
                break

            action_json = choose_action(model, tokenizer, state_str, valid_actions, args)
            state_str, reward, done, info = env.step(action_json)
            total_reward += float(reward)
            print(
                f"step={step_idx} turn={info.get('turn')} action={action_json} "
                f"reward={reward:.3f} total_reward={total_reward:.3f} done={done}",
                flush=True,
            )

        final_battle = env._ensure_battle()
        print(
            f"Battle complete | won={getattr(final_battle, 'won', None)} "
            f"lost={getattr(final_battle, 'lost', None)} total_reward={total_reward:.3f}",
            flush=True,
        )
        print(f"Room URL: {room_url}", flush=True)
    finally:
        if showdown_proc is not None and showdown_proc.poll() is None:
            showdown_proc.terminate()


if __name__ == "__main__":
    main()
