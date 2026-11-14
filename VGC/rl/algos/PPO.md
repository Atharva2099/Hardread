# PPO, read as "REINFORCE + fixes"

`ppo.py` is one file in CleanRL's layout. Same networks, masking and checkpoint format as
`reinforce.py`; what changes is how the gradient is estimated and how often we may use it.

## Each piece fixes a REINFORCE weakness

| REINFORCE weakness | PPO fix | Where in `ppo.py` |
|---|---|---|
| Every step of a ~65-step game gets credit for the final +-1 (very noisy) | a **critic** V(s) and **GAE**: judge each step by what it changed | `compute_gae` |
| One gradient step, then the games are thrown away (sample-hungry) | fixed-length rollout, then **4 epochs x 4 minibatches** on it | `update` |
| Nothing limits the step size; one bad step can wreck the policy | **clipped ratio** keeps the new policy close to the one that collected the data | `ppo_loss` |

## The maths in plain terms

* **TD error** `delta_t = r_t + gamma V(s_t+1) - V(s_t)`: "what I got plus what I now expect, minus
  what I expected". The critic's guess replaces the unknown rest of the game (*bootstrapping*): lower
  noise, some bias if the critic is wrong. At the end of a game the next value is 0.
* **GAE** `A_t = delta_t + gamma lambda A_t+1`: blends one-step estimates (lambda 0: low noise, biased) and
  the full-game return (lambda 1: unbiased, noisy). We use 0.95. Advantage > 0 means "better than the
  critic expected", so make that action likelier.
* **Ratio** `r = pi_new(a|s) / pi_old(a|s)`, computed from log-probs. It is 1 on the first minibatch and
  drifts as we re-use the data.
* **Clipped objective** `min(r A, clip(r, 0.8, 1.2) A)`: once r leaves [0.8, 1.2] in the direction the
  advantage likes, the gradient is 0, so there is no reason to push further.
* **Value loss** `0.5 (V(s) - (A + V_old(s)))^2`: train the critic towards the GAE return.
* **Advantage normalisation** (per minibatch) makes the step size independent of the reward scale.
* **Entropy bonus** (0.01) keeps some exploration; **grad-norm clip** (0.5) and **linear lr decay** keep
  updates tame. Illegal actions are masked; steps whose action the server refused (`valid=False`) get zero
  weight in every loss.
* **Done handling.** The env auto-resets, so the obs after a finished game belongs to a new one: a
  terminal step (win, loss, tie, or 300-turn cap) uses next value 0 and cuts the GAE chain. Only games
  still running when the rollout ends are bootstrapped from V(next_obs).

## Metrics and healthy ranges

| metric | meaning | healthy |
|---|---|---|
| `approx_kl` | how far the policy moved from the data-collecting one | about 0.01-0.03 (much lower: lr may be too small; much higher: too large) |
| `clipfrac` | share of steps whose ratio was clipped | about 0.1-0.3 (0 means clipping never matters) |
| `explained_variance` | `1 - Var(return - V) / Var(return)`: is the critic useful? | rising toward > 0.3; <= 0 means it is no better than a constant |
| `entropy` | undecidedness of the policy (max ln 9 = 2.2 over 9 actions) | slowly falling; a collapse to ~0 early is bad |
| `policy_loss`, `value_loss` | for debugging only; the policy loss hovers near 0 by construction | value loss should fall |
| `grad_norm` | pre-clip gradient norm | below ~0.5 = clipping rarely active |

With the default lr (2.5e-4) and few updates the early `approx_kl` is far below 0.01: the policy is moving
slowly; raise `--lr` or `--update-epochs` if that persists once the critic is useful.

## Curriculum

Opponents: RandomPlayer, then MaxBasePowerPlayer, then SimpleHeuristicsPlayer. A phase advances when the
win rate over the last 200 games against its opponent is >= 0.70 and at least 300 games were played
in the phase. After the first advance, one of the 4 envs keeps playing an earlier opponent (cycling
every 10 updates) so the policy does not forget how to beat them. A game in progress always finishes
against its old opponent; the switch applies at that env's next game. Look at the per-opponent win rates
(`win_rate_by_opp` in `metrics.jsonl`, plotted by `python -m rl.plot_metrics <run_dir>`; red lines =
advances). `episodes.jsonl` lists the opponent of every finished game.

Run: `uv run --extra bench --extra rl python -m rl.algos.ppo --run-id ppo1` (add `--no-curriculum
--opponent MaxBasePowerPlayer` for a fixed opponent).

## Crash-proofing and resuming

A 2M-step run once died because two games stalled for 60 s on a machine that was out of memory.
One slow game must never kill a run, so:

* **Stalls.** If a game gets no decision from the server for `--stall-timeout` seconds (default 60),
  the env worker abandons it (best-effort forfeit), closes and rebuilds its `PokemonEnv` (same port,
  current opponent) and reports the lost game as a terminal step with `truncated=True`, reward 0 and
  `info["episode"] = {"aborted": True, "reason": "stall"}`.  PPO treats that as **no outcome**: the
  step is `done` (GAE is cut) and `valid=False` (zero weight), and the game is in no win-rate or
  curriculum statistic.  The game's earlier steps stay in the batch (their advantages are cut at the
  abort; one slightly biased step per rare abort).  REINFORCE drops the whole aborted episode.
  `n_aborted` is logged per update in `metrics.jsonl` (and aborted games in `episodes.jsonl` with
  `"aborted": true`).  More than `--max-aborts` (20) in a row in one worker, or an env that cannot be
  rebuilt in 3 tries, raises.
* **Showdown dying.** If the shared server process has exited when workers report failure, the
  parent restarts it (on a new free port), respawns all workers and reports an aborted step
  (`reason="server_restart"`) for every env.  3 restarts, then it raises.
* **Resume.** `latest.pt` (every `--save-every` updates, default 10, written atomically) holds the nets,
  optimiser, counters, the full curriculum state, RNG states and args.
  `python -m rl.algos.ppo --resume outputs/rl/<run_id>` continues at the next update (LR anneal
  relative to the ORIGINAL `--total-steps`), appends to `metrics.jsonl`/`episodes.jsonl` (rows written
  after the checkpoint are dropped first, so there are no duplicates) and writes a `resumed` line to
  `events.jsonl`.  Refuses a different `OBS_VERSION`.  Games in flight at the crash are lost.  Only
  `--stall-timeout/--max-aborts/--save-every/--ckpt-every` can be changed on resume.
* **Supervisor.** `rl/run_forever.sh <run_id> [ppo args]` restarts a crashed run from `latest.pt`
  (`MAX_RESTARTS`, default 5; 30 s wait; log in `<run_dir>/supervisor.log`).
