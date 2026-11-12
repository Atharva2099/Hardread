# TCG Data Quality Audit — Phase 1/2

Date: 2026-08-09
Source: private Kaggle dataset `atharvawalawalkar/hardread-tcg-bc-pairs` (111 NPZ shards)

## Inventory

- 494,215 extracted state/action pairs
- 7,270 episode IDs
- 323,280 winner transitions (65.4%)
- 170,935 loser transitions (34.6%)
- Deck labels: Lucario 95,932 (19.4%), Crustle 30,666 (6.2%), Alakazam 236,252 (47.8%), unknown 131,365 (26.6%)
- Episode pair count: median 62; 1st–99th percentile 6–218; maximum 402

## Findings

1. **Deck imbalance is severe.** Alakazam is 47.8% of pairs while Crustle is 6.2%. Unknown is 26.6%, so deck-conditioned training currently receives noisy/incomplete labels.
2. **The archive lacks the metadata needed for the intended quality filter.** NPZ files retain outcome, episode ID, and recovered deck, but not agent name, opponent name, Elo, timeout/error status, or raw replay provenance. Elo filtering cannot be reproduced from this archive.
3. **Episode extraction is asymmetric.** Only 498/7,270 episodes (6.8%) contain both outcome signs; 6,772 episodes contain only winner or only loser transitions. This must be explained before treating outcome as a reliable transition-level quality signal.
4. **Action-count invariant is violated.** 15,865 pairs (3.21%) have `chosen_count > max_count`, mostly where `max_count == 1`. These pairs are quarantined conceptually but not silently deleted; simulator/schema semantics must be checked first.
5. **Single-action labels dominate.** 307,362 pairs (62.2%) are select type 0; 473,135 (95.7%) are single-action labels. We must verify whether select type 0 means forced choice; regardless, raw validation accuracy will be dominated by single-action decisions.

## Phase-2 dataset variants created

`experiments/dataset_variants_v1.json` defines reproducible episode-level variants:

- `all_valid`: baseline archive.
- `deck_balanced`: inverse-frequency episode weighting by recovered deck.
- `both_outcomes_holdout`: 498 episodes for diagnostics only, not primary training.
- `nontrivial_episodes`: 6,751 episodes with at least 20 pairs.

These are manifests/weights, not silently rewritten training shards.

## Decision

Do **not** train a new BC model from this archive yet. The next data task is to regenerate or retrieve raw replay metadata, then validate the `chosen_count/max_count` mismatch and recover agent/opponent identity. Only after that should we apply Elo/opponent-quality filtering.
