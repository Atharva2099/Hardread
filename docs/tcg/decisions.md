# Decisions Log

Running log of design decisions, gotchas, and rule clarifications. Add entries at the top.

---

## cabt vs Official TCG — Simulator is Source of Truth

Source: [Kaggle discussion #708586](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586). In this competition, **simulator behavior is treated as correct**, even where it diverges from official Pokémon TCG rules.

### 1. Unresolvable attacks are not selectable

- **Official:** Player may declare an attack whose effect cannot fully resolve; turn simply ends.
- **Sim:** Same attack is not selectable from the start.
- **Examples:**
  - Attack that puts a Basic Pokémon from deck onto Bench when bench is full.
  - Attack that draws cards when deck is empty.
  - Attack that interacts with opponent's hand when opponent's hand is empty.
- **Impact:** Minimal — end result is the same.
- **Action for our agent:** When reasoning about legality, rely on the sim's selectable set, not official-rule legality. The sim never offers these attacks, so we never need to handle "declared but failed" cases.

### 2. Mega Zygarde ex — Nullifying Zero target order

- **Official:** Attacking player chooses the order in which damage is assigned across targets.
- **Sim:** Coins are flipped automatically left-to-right; no choice.
- **Impact:** None — Knock Out processing is simultaneous in both.
- **Action for our agent:** Don't model target-order choice for Nullifying Zero. Skip Mega Zygarde ex as a deck choice unless we want to test coin-flip variance.

### 3. Prize-taking order on simultaneous KOs

- **Official:** Both players choose their prizes, then both take simultaneously. Then next-turn player promotes a new Active.
- **Sim:** Next-turn player chooses AND takes their prizes, then opponent chooses AND takes theirs. Then next-turn player promotes.
- **Impact:** None in this competition — full prize clear by both = draw, regardless of order.
- **Action for our agent:** Don't optimize for prize-order strategy. It's a draw either way.

### General rule

**Trust the sim.** When in doubt, the sim is correct for this competition. Our agent should be tested against the sim's actual behavior, not against our mental model of official TCG rules.
