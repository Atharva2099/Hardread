# REINFORCE, as implemented in `reinforce.py`

## The idea in plain language
Play some games with the current policy. Look at how each game ended. If it was a win, make
*every action taken in that game* a little more likely; if a loss, a little less likely.
Do this thousands of times and the actions that tend to occur in wins become common. No model of
the game, no search: just "do more of what preceded good outcomes". It is noisy, because a win
also contains bad moves and a loss good ones; averaging over many games is what filters that out.

## The math
The policy pi(a|s) is the softmax of the network's 9 logits. We want to maximise J = E[G], the
expected return. The policy-gradient theorem says

    grad J = E[ sum_t  grad log pi(a_t | s_t) * G_t ]

*Why log-prob x return?* `grad log pi(a|s)` is the direction in parameter space that makes action
`a` more likely in state `s`; multiplying by G_t scales (and signs) that push by how well things
went afterwards. We cannot differentiate "the outcome of a game", but we do not need to: sampling
games and weighting the log-prob gradients gives an unbiased estimate of grad J. In code
(`update`, `policy_loss = -(logp * adv).mean()`) we write a loss whose gradient equals that
expression; the minus is because optimisers minimise.

Returns (`compute_returns`): G_t = r_t + gamma * G_{t+1}, filled in backwards from the last step.
Reward is only +1/-1/0 on the final step, so G_t = gamma^(T-t) * R_T. A truncated game (300-turn
cap) is treated as terminal: nothing is bootstrapped.

## The baseline
Replace G_t by A_t = G_t - V(s_t). Subtracting anything that depends only on the state does not
change the expected gradient (it only moves the weights of actions within a state), but it can
cut the variance a lot: with raw G, a state where we usually win pushes up *all* its actions,
good or bad. With A, only actions that did *better than usual from that state* are pushed up.
V(s) is a second, separate network trained by regression (`value_loss = MSE(V(s), G)`); it is
`.detach()`ed in the policy loss so the policy loss never trains it. `--baseline none` uses A = G.
`--norm-adv` additionally rescales A to mean 0 / std 1 per batch (off by default).

## Masking
Illegal actions (fainted switch targets, a forced switch, ...) get logit -1e9 before the softmax
(`masked_dist`). They get probability exactly 0, so they are never sampled, add nothing to the
entropy, and get no gradient. The policy is renormalised over legal actions only, and the same
distribution is used when sampling and when computing log pi, so the gradient is exact.

## Entropy bonus
`-ent_coef * entropy` is added to the loss to keep the policy from collapsing onto one action
before it has seen enough evidence. It is small (0.01); too large and the policy stays random.

## Reading the log (`metrics.jsonl`, one row per update)
| metric | meaning | healthy | broken |
|---|---|---|---|
| `win_rate`, `rolling_win_rate` | wins in this batch / last 100 episodes (ties count as not a win). Random vs random is ~0.5 | slowly drifts up, noisy (16 episodes = +-0.12 per batch) | flat at 0.5 after thousands of episodes, or falling |
| `entropy` | policy uncertainty; ln(#legal) at start (~1.8 here) | decays gradually | collapses to ~0 early (premature determinism), or stays at its start value (not learning / ent_coef too big) |
| `policy_loss` | -mean(logp * A). Not a performance measure: it hovers around 0 and bounces with the batch's outcomes | small, noisy, no trend | NaN, or huge |
| `value_loss` | MSE of V(s) vs G. With +-1 outcomes and no skill the best constant gives ~1 and a useful V gets below | falls from ~0.5-1 and stays below the variance of G | rising or NaN |
| `grad_norm` | pre-clip gradient norm of the policy (clip at 0.5) | well under the clip, noisy | always at the clip (steps are being truncated), or ~0 (dead net) |
| `mean_abs_advantage` | size of A. With a good baseline it is smaller than without | below the |G| level | not shrinking = V is not learning |
| `mean_return` | average G over the batch's steps | tracks the win rate | |
| `mean_ep_len`, `steps_per_sec` | game length in our decisions; throughput | ~60-70 steps | length pinned near the turn cap |

Runs are not bit-reproducible: `--seed` seeds torch/numpy/random, but Showdown draws random teams.
