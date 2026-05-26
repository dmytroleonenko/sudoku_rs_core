# GFlowNet Pilot — Specification

**Status:** Spec v1 PAUSED — see §15 (double-review verdict 2026-05-20: FATAL on MDP).
Spec v2 pending after vicinity-climb empirical baseline (Task #7) lands.
**Goal:** validate whether a Trajectory-Balance GFlowNet can sample BxB ≥ 6 Sudoku puzzles proportional to reward, after the UA-constructor path failed empirically (round 1 + round 2, see `research/hardest_puzzle_construction_notes.md` §6–§7).
**Owner:** Opus subagent for implementation (per CLAUDE.md user instruction).
**Runtime:** MLX on M3 Max (Apple Silicon). JAX-METAL is empirically broken; do NOT use.

## 1. Problem statement

Sample puzzles P from a learned distribution π(P) ∝ R(P), where
- `R(P) = exp(α · max(0, BxB(P))) · 𝟙[is_unique(P)] · 𝟙[clues(P) ∈ [20,28]]`
- BxB computed by `shc::te::rate_bxb(P, max_length=14, buffer_size=1M)` via FFI/CLI.
- α = 1.5 (tunable; gives R(BxB=6) ≈ 8100, R(BxB=3) ≈ 90, R(BxB=0) = 1).

Pilot validates whether GFlowNet's mode-coverage property (Bengio et al. 2021, 2022) yields diverse BxB ≥ 6 sampling at a hit rate that justifies the full ML pipeline. Decision after ≤ 8 GPU-hours on M3 Max.

## 2. MDP construction

**State `s`:** partial Sudoku P with k clues placed (0 ≤ k ≤ 30). Encoded as 81 × 10 one-hot (digit 0 = empty).
**Initial state `s₀`:** a random full solution grid G (cached per-trajectory); P starts empty.
**Action space `A(s)`:** at step k, the agent picks one of the 81−k empty cells and reveals G at that cell. Action = cell index ∈ {0..80} (digit determined by G).
**Terminal condition:** k = K (sampled from {20..28} via a learned `K` head, or fixed = 24 in pilot).
**Reward `R(sₜ)`:** only at terminal; computed by Rust CLI subprocess `rate-shc` returning u16 BxB and uniqueness bool.

**Forward policy `P_F(a|s; θ)`:** softmax over 81 cell logits, masked to empty cells.
**Backward policy `P_B(s|s'; θ)`:** uniform over revealed cells (clue-removal) for the pilot — avoids learning a separate net.

## 3. Acceptance criteria

After 8 GPU-hours of training + 1k-sample final evaluation:

**PASS — proceed to full version:**
- ≥ 5% of 1k final-eval samples have BxB ≥ 6.
- ≥ 30% have BxB ≥ 3 (mode-coverage sanity).
- Loss trajectory converges (TB loss decreasing, < 1.0 average over last 1k steps).
- Distinct grids covered: ≥ 100 unique BxB ≥ 5 puzzles in eval.

**MARGINAL — extend training:**
- BxB ≥ 6 hit rate ∈ [0.5%, 5%]. Mode-coverage partial. Investigate temperature, α tuning, extend to 24 GPU-hours.
- Loss decreasing but plateau before convergence.

**FAIL — pivot:**
- BxB ≥ 6 hit rate < 0.5% AND BxB ≥ 3 < 10%. Net is not learning the reward gradient.
- TB loss non-decreasing or NaN.
- ≥ 20% of eval samples non-unique.

## 4. Architecture

```python
# tools/sudoku_rs_core/python/gflownet_pilot/model.py (MLX)

class GFlowNetPolicy(nn.Module):
    def __init__(self, d=128, n_layers=4):
        self.embed = nn.Linear(10, d)        # cell one-hot → d
        self.pos_embed = nn.Embedding(81, d) # cell position
        self.blocks = [TransformerBlock(d, n_heads=4) for _ in range(n_layers)]
        self.head_logits = nn.Linear(d, 1)   # per-cell action logit
        self.log_Z = mx.zeros(1)             # learned partition

    def __call__(self, state):  # state: (B, 81, 10)
        x = self.embed(state) + self.pos_embed(mx.arange(81))
        for blk in self.blocks: x = blk(x)
        return self.head_logits(x).squeeze(-1)  # (B, 81)
```

~150k params; ≤ 50MB MLX memory. Forward pass < 5ms on M3 Max per state.

## 5. Trajectory Balance loss

Standard TB (Malkin et al. 2022):
```
L_TB(τ) = (log Z + Σₜ log P_F(aₜ|sₜ) - log R(s_T) - Σₜ log P_B(sₜ|sₜ₊₁))²
```
For uniform `P_B` over k revealed cells: log P_B = -log(k) per step.
Total per-trajectory loss is a single squared scalar; `log_Z` is a learned parameter.

## 6. Training loop

```
for step in range(N_STEPS):              # ~50k–200k
    sample G ~ random_full_grid()
    rollout K cells under P_F (greedy + temperature τ=2.0 → 1.0 annealed)
    P = build_puzzle(G, revealed)
    R = compute_reward_via_rust_cli(P)  # subprocess, batched async
    loss = TB(trajectory, R)
    loss.backward(); optimizer.step()
```

**Reward computation pipeline:** spawn pool of N=8 `sudoku_rs_core rate-shc-batch --classification bxb` workers; trajectories write their puzzle to a queue; workers return `(puzzle_id, bxb, is_unique)` JSONL. Per-puzzle wall median 5s; batch 32 → ~20s wall per batch.

**Replay buffer:** prioritized by R; sample 50% from buffer, 50% on-policy fresh.
**Batch size:** 32 trajectories (training); 128 (eval).

## 7. CLI / runtime

```bash
# Local M3 Max
cd tools/sudoku_rs_core/python/gflownet_pilot
uv run python train.py \
    --max-steps 100000 \
    --target-bxb 6 \
    --alpha 1.5 \
    --K 24 \
    --reward-workers 8 \
    --rust-bin ../../target/release/sudoku_rs_core \
    --output-dir runs/gfn_pilot_v1 \
    --eval-every 5000 \
    --final-eval-n 1000
```

Outputs:
- `runs/gfn_pilot_v1/train.jsonl` — per-step loss, log_Z, avg R, BxB histogram.
- `runs/gfn_pilot_v1/eval_*.jsonl` — periodic 256-sample evals.
- `runs/gfn_pilot_v1/final_eval.jsonl` — 1000-sample post-training eval.
- `runs/gfn_pilot_v1/ckpts/` — MLX weights every 10k steps.

## 8. Testing

- `test_random_grid_generation`: full-grid generator returns valid 81-cell Sudoku.
- `test_rust_cli_roundtrip`: known forum_hardest puzzle returns expected BxB rating from subprocess.
- `test_tb_loss_zero_at_optimum`: hand-construct trajectory where log_Z + log P_F − log R − log P_B = 0; verify loss = 0.
- `test_action_mask`: revealed-cell mask correctly excludes already-placed clues.
- `test_smoke_50steps`: run 50 training steps; verify loss decreases and no NaN.

## 9. Risks (descending probability)

1. **Reward computation bottleneck.** `rate_bxb` median 5s; 100k trajectories × 5s = 140 GPU-hours just on reward. Mitigation: hard cap `--rate-bxb-timeout-ms 8000`, downgrade to BxB ceiling (return -3 = "exceeds buffer"); pre-flight benchmark over 100 random grids.
2. **Sparse reward.** Random rollout will yield BxB=0 puzzles ~99% of the time initially. Mitigation: warmup with reward shaping — give partial credit for unique non-trivial puzzles (clues ∈ [20,28] AND unique → R_floor=10).
3. **Mode collapse.** TB is mode-covering in theory but in practice can collapse without enough exploration. Mitigation: KL-from-uniform regularizer (β=0.01), temperature annealing 2.0 → 1.0.
4. **Non-uniqueness epidemic.** Without UA structure, randomly revealing 24 clues from a full grid → unique ~30% of the time. Mitigation: include uniqueness as binary multiplicative reward; consider increasing K to 26–28 mid-training.
5. **MLX maturity gaps.** Some ops (e.g. masked softmax + sampling) may need manual implementation. Mitigation: keep architecture minimal (transformer + linear); avoid exotic primitives.
6. **FFI overhead.** Subprocess spawning at 5s/call dominates wall if reward workers are starved. Mitigation: keep persistent worker processes via stdin/stdout streaming protocol.

## 10. Stop / continue decision

After 8 GPU-hours OR 100k training steps (whichever first):
- **PASS** criteria from §3 → write up; commit to full version (replay buffer + larger net + multi-K head + ILP-style refinement post-sample).
- **MARGINAL** → extend to 24 GPU-hours with τ schedule tweak + α grid {1.0, 1.5, 2.0}.
- **FAIL** → publish negative finding to `research/hardest_puzzle_construction_notes.md` §8; pivot to (a) ES on a parametric puzzle generator, (b) supervised distillation from forum_hardest corpus, or (c) accept community-curation as the only viable path for BxB ≥ 6.

## 11. Out of scope

- Backward policy learning (uniform P_B suffices for pilot).
- Multi-K head (fixed K=24).
- Distributed training (single M3 Max).
- ILP refinement post-sample.
- 16×16 generalization.
- Adversarial training against rate_bxb.
- Comparison with PPO / SAC baselines (deferred to full version if PASS).

## 12. Deliverables

1. `python/gflownet_pilot/{model.py, train.py, reward.py, env.py}` — MLX implementation, ~600–900 LOC.
2. `python/gflownet_pilot/tests/` — 5 tests from §8, ~150 LOC.
3. Rust changes: stdin-streaming mode for `rate-shc-batch` (persistent worker), ~100 LOC in `src/cli.rs`.
4. Pilot run JSONL (train + eval) + final-eval summary committed under `runs/gfn_pilot_v1/`.
5. Verdict + analysis (~400 words) appended to `research/hardest_puzzle_construction_notes.md` §8 (NEW).

## 13. Verification before launch

- `cargo build --release` clean (Rust workers).
- `uv run pytest python/gflownet_pilot/tests` all green.
- `train.py --max-steps 50 --reward-workers 2` smoke run completes < 10 min.
- Subprocess pipeline benchmark: 100 random puzzles through `rate-shc-batch --stdin` → median ≤ 5s confirmed.
- Manifest commits clean git SHA + dataset hash (random-grid seed range fixed).

## 14. Kill-switch criteria (during run)

Auto-abort training if any of:
- TB loss = NaN for 100 consecutive steps.
- Eval BxB ≥ 6 rate = 0 at step 20k AND step 40k (no learning signal).
- Reward-worker queue depth > 1000 for 5 minutes (compute-starved; restart with more workers).
- M3 Max sustained > 90°C for 10 min (thermal protection).

Watcher script: `scripts/gfn_pilot_kill_watcher.py`, polled from launching tmux session.

## 15. Double-review verdict 2026-05-20 — v1 PAUSED

Both Claude reviewer (NEEDS-FIX) and Codex reviewer (FATAL) converged on one fatal architectural flaw plus several serious issues. Spec v1 not launched.

### Fatal flaw
**MDP action space is structurally equivalent to UA hitting-set selection.** State = (random full grid G, revealed-cells); action = pick next cell to reveal; terminal puzzle is determined entirely by which K cells of G are kept as clues. This is exactly the space UA-constructor pilot already proved empty of BxB≥6 puzzles in 2 empirical rounds (0/100 hits, pre-repair uniqueness 0%). No learned policy can find modes that don't exist in the sampled space.

### Critical issues (both reviewers)
- Reward budget arithmetic: 100k steps × 5s / 8 workers ≈ 17 GPU-h, not the spec's 8 GPU-h.
- Timeout truncation (`--rate-bxb-timeout-ms 8000`) censors exactly the tail where BxB≥6 lives — reward oracle biased against the target mode.
- PASS criterion 5% BxB≥6 in 1k samples is uncalibrated vs community baseline 0.16% (79/48766 in forum_hardest).
- R_floor=10 for unique puzzles ∝-shifts mass toward easy-unique mode, doesn't fix sparse-reward pathology of TB.
- Uniform `P_B` correct only for fixed K and unordered subset terminal — variable-K head (mentioned §2) breaks TB derivation without explicit stop-action.
- MLX maturity on masked categorical sampling + async reward plumbing treated as minor risk; should be proven before any transformer buildout.

### Plan
1. **Run vicinity-climb baseline first (Task #7).** `gen-vicinity --fitness bxb --target-bxb 6` from the 1166 forum_hardest BxB=5 seeds is already shipped and cheap. It empirically calibrates whether the *seed-vicinity* space (single-cell mutations from a known BxB=5 puzzle) contains BxB≥6 modes at all.
2. If vicinity yields ≥1 BxB≥6 hit: ML may be unnecessary — escalate vicinity to larger neighbourhood depth, ship as the constructor.
3. If vicinity yields 0 hits: redesign GFlowNet MDP as **seed-conditioned edit-and-repair** in puzzle-mutation space (swap cells, flip clue/non-clue, perturb digit pair) starting from BxB=5 seeds, not random grids; use SubTB instead of TB; pretrain on cheap reward proxy (B-rating, ~1ms) before BxB. Re-spec as v2 and re-review.
