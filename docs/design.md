# cancho-rl: reinforcement learning on cancho-gpu (a design, written before the code)

> Written 2026-11-04, before any code. **Nothing in this document is built or measured in this repository.** Claims carry a mark: **[read]** from a file of another repository (named), **[derived]** from numbers that were measured elsewhere, **[estimate]** a guess with its reasoning, **[unknown]** not looked at yet and to be read before the milestone that depends on it. A claim that turns out false is corrected here, in place, with the date.

## 1. What this is, and what it is not

`cancho-rl` is where reinforcement-learning *algorithms* live: the rollout loop, rewards, advantage estimators, the policy-gradient loss and the experiments that measure them. It runs on top of [`cancho-gpu`](https://github.com/alpibrusl/cancho-gpu), which owns the kernels, the training sessions and the runners.

The split, and why:

* **cancho-gpu keeps the primitives.** A kernel, a session directive, a faster decoder, a numerically checked loss gradient. Its gate holds numerics to 1e-4 against torch and its documents are about the training stack; algorithms that change every week do not belong in it.
* **cancho-rl keeps the algorithms and the experiments.** GRPO, REINFORCE with a leave-one-out baseline, PPO if ever; reward functions; the evaluation protocol.
* **The environments stay where they live.** Tasks with a checkable reward for a coding model come from `cancho-code` (its eval harness and journals); a robot policy would come from `cancho-robot`'s simulator. `cancho-rl` calls them; they do not depend on it.

Not goals: a general RL framework, a gym replacement, multi-GPU or NVIDIA training (cancho-gpu's CUDA path has run only kernels so far, not a training session), RLHF with a learned reward model.

## 2. The first target, and why this one

**GRPO** (group-relative policy optimisation: sample a group of completions per prompt, score each with a reward, use the reward minus the group mean as the advantage, no value network) with **LoRA** on a small Qwen (Qwen2.5-0.5B and 1.5B are the sizes cancho-gpu trains today), on **tasks with a verifiable reward from cancho-code**: write or repair a cancho program, reward from the compiler's check and the task's tests.

Why this combination:

1. No critic network: memory is the policy plus its adapters. cancho-gpu's 1.5B LoRA step is 19.4 GB at 4 x 512 tokens **[read: cancho-gpu `docs/resident.md` section 23]**; a second model would not fit comfortably next to rollouts on 64 GB.
2. **The reference policy is free with LoRA**: the same base with the adapter switched off gives the KL term's reference log-probabilities **[unknown: whether a session can run one forward with the adapters off; to be read in `examples/qwen/qwen_session.py` before R2]**.
3. The reward is checkable without a learned model, which keeps the experiment honest and cheap. It is also the place where the experiment can fool itself (section 7).
4. cancho-code's M8 experiment already asks "does tuning on cancho help the coding agent?" (cancho-gpu `docs/roadmap-models.md` section 7). RL on the same tasks is its natural next question, with a measured baseline from M8's arms.

## 3. What cancho-gpu already has, and what is missing for RL

| need | state | source |
|---|---|---|
| LoRA forward, backward and AdamW on Qwen2.5-0.5B / 1.5B, adapters equal to peft's | built, measured (gradients cosine >= 0.9994 against peft, 150-step runs within 4e-4 of peft's loss) | **[read]** `docs/resident.md` sections 21-24 |
| 4-bit base (NVFP4) with LoRA, 1.5B and 7B | built | **[read]** sections 27, 28 |
| A per-row weight on the cross-entropy gradient | **exists**: `ce_grad.chx` takes a weight `wt` per row (the loss scale, 0 for the last position of a sequence) | **[read]** `examples/qwen/ce_grad.chx` |
| An advantage-weighted loss (`-A_i * log pi(token)` over the completion tokens) | **nearly there**: it is the same row weight with the advantage folded in, and zero on the prompt tokens (the M8 training already masks prompt tokens) | **[derived]**; the exact kernel contract is **[unknown]** until read |
| The log-probability of the sampled tokens (for the ratio and for logging) | `ce_stats` / `ce_comb` produce per-row loss, which is the negative log-probability of the label | **[read]** `ce_comb.chx` |
| Generation with a KV cache **in the training session of Qwen2.5** | **missing**: `generate.py` recomputes the window and starts a new session per token, 0.9 s a token for the 0.5B | **[read]** `docs/resident.md` section 22.5 |
| A cached decoder | exists for **Qwen3.5-27B** (36.5 ms a token, one request at a time, a 1,024-position cache) | **[read]** `docs/qwen35.md`, `docs/serving.md` |
| Batched sampling of a group of completions | **missing**: the runners serve one request at a time | **[read]** `docs/serving.md` |
| Moving updated adapters into the generator | merging into f16 weights is done offline by `generate.py` | **[read]** section 22.5; its cost per iteration is **[unknown]** |
| Rewards from `cancho check` / tests, run in a sandbox | in cancho-code's harness | **[unknown]**: its API and cost to be read before R0 |

**The decisive gap is rollouts.** A group of 8 completions of 512 tokens on the 0.5B is 512 decode steps of a batch of 8; at the measured 0.9 s a token that is about 460 s for one group **[derived]**, against a gradient step of about 0.9 s for 4,096 tokens **[read]**. An RL loop whose rollouts cost 500 times its learning step cannot be measured, let alone be fast. So **the order of the milestones is rollouts first** (R3 before R4), and the first working loop (R1) uses a different generator.

## 4. Milestones, each ending at a number

Exit criteria are fixed here, before building. Each milestone is a separate change with a gate that can fail.

**R0. A task set and a reward that can be trusted.** From cancho-code's tasks, a set where the untuned model sometimes succeeds and sometimes fails, so that there is a signal to learn from. Exit: (a) the reward is deterministic (the same program gives the same reward in 20 repeated runs, 0 disagreements); (b) on at least 200 tasks, split into train and held-out with the held-out tests never shown to any policy, the base model's pass rate with 8 samples per task is between 10% and 70% (outside that range there is nothing to learn from or nothing to measure); (c) a list of reward-hacking probes (an empty program, a program that prints the expected output, a program that exits 0 early) each scoring zero. Baselines recorded: the base model and, if M8 has finished, the M8 arms.

**R1. A reference loop in torch/peft on MPS (the oracle).** The same discipline as cancho-gpu's M7: first the established library, so that every later number has something to equal. GRPO with peft LoRA, torch on MPS, generation by the model's own `generate` or by MLX, rewards from R0. Exit: the held-out pass rate after N iterations is above the base model's by more than the spread of two seeds, with the seeds, the iteration count and the wall time recorded. If it does not rise, that is the result, and the milestones after it are not built on hope.

**R2. The policy-gradient step in a cancho-gpu session equals torch's.** One fixed batch of groups: the advantage-weighted loss and the gradient of every adapter against R1's, as cancho-gpu's M7.2 did. Exit: the loss within 1e-3 relative, every adapter gradient at cosine >= 0.995, a 40-step run whose training loss differs from torch's by under 1e-3 (the M7.3 pattern). Needs: the weighted loss kernel contract read (section 3), the reference forward with adapters off.

**R3. Rollouts at a usable speed.** A cached batched decoder for the Qwen2.5 sessions (or the existing runner extended), able to take freshly updated adapters. Exit: greedy tokens equal the R1 generator's on a fixed set of 8 prompts for 128 tokens (the M2 and M7.3 standard), and tokens a second for a batch of 8 completions reported against MLX and llama.cpp on the same model. **No target speed is set here**: the number is measured first, then the claim is made, because the competitors on this machine are fast and the honest question is whether cancho-gpu is within reach, not whether it wins.

**R4. The loop end to end on cancho-gpu.** R1's experiment with R2's step and R3's rollouts. Exit: the held-out pass-rate curve within the spread of two R1 seeds at every checkpoint, and seconds per iteration and peak memory reported against R1's. If the rollouts are still the cost (they will be, section 6), the report says what fraction of an iteration they take.

## 5. The experiment, stated before it is run

* **Model:** Qwen2.5-1.5B-Instruct is the first choice (the size M8 tuned and cancho-gpu trains in 19.4 GB); the 0.5B for fast iteration of the code. **[open]** the maintainer's call.
* **Algorithm:** GRPO with the advantage `A_i = (r_i - mean(r)) / (std(r) + eps)` over a group of 8, a KL penalty to the reference with a small coefficient, one gradient step per batch of groups (no PPO clipping at first: it matters only with several steps on the same samples). Learning rate, group size and KL coefficient are chosen on the *train* tasks and fixed before the held-out tasks are touched.
* **Seeds:** two at least for R1; fewer is not an experiment.
* **Success rule for "RL helps":** the held-out pass rate at the end is above the starting rate by more than the interval of the starting rate itself (a bootstrap over tasks, 10,000 resamples, a seed recorded), in both seeds. "Helps more than M8's supervised tuning" is a second, separate claim that needs the M8 result and the same held-out tasks.
* **What is reported whether or not it works:** the reward curve on train tasks, the pass rate on held-out tasks, the length of the completions over time (length collapse and loops are the usual failure; cancho-gpu's W0 repeat detector `examples/qwen/m8_loops.py` is a ready check for loops **[read]**), and the fraction of groups with zero reward variance (those give zero gradient and measure how hard the task set is).

## 6. Cost and where the time goes

* **Rollouts dominate** every RL loop of this kind; here more than usual, because the cached decoder for the training model does not exist yet (section 3). Until R3, an iteration is rollout-bound by orders of magnitude **[derived]**.
* **Memory:** policy weights (f16, 3 GB at 1.5B), adapters and Adam (small), activations for the gradient step (the 19.4 GB figure includes them), plus the generator's KV cache. Rollouts and the gradient step do not run at the same time in the first design, so the peaks do not add **[estimate]**; one GPU job at a time on this machine is a standing rule of the workspace.
* **R1 on torch/MPS** is the cheapest way to learn whether the *task set* gives a signal. It should be run before any cancho-gpu work, because it can end the project early at no kernel cost.

## 7. Risks, said now

1. **Reward hacking.** A checker that accepts "compiles" rewards trivial programs; a test the policy can see is a test it can memorise. R0(c) and the never-shown held-out tests are the defence, not a guarantee.
2. **No signal.** If the base pass rate is near 0 or near 1 on the task set, GRPO has nothing to learn from. R0(b) exists to find that out first.
3. **Small-model RL is noisy.** Two seeds can disagree. The success rule requires both; if they disagree the result is "not shown".
4. **Rollout speed decides feasibility, not the algorithm.** If R3 lands far from the alternatives on this machine, the right move may be to use MLX or llama.cpp for rollouts and keep cancho-gpu for the gradient step (R2 with R1's generator). That is a legitimate end state and is recorded as one.
5. **The M8 training this builds on was torch/MPS, not cancho-gpu** (its memory growth swapped the machine before it was fixed). Comparisons of tuned-model baselines must say which stack trained them.
6. **Everything measured so far is one Apple M4 Max.** No NVIDIA training run exists; nothing here claims one.

## 8. Working rules for this repository

Inherited from cancho-gpu and cancho, because they work:

* **Design before code**, in `docs/`, with claims marked as in this document. A claim that turns out false is corrected in place, in the document that made it.
* **The gate:** whatever the repository's gate script runs must pass before anything is called done; a test that cannot fail is not a test (each check is watched failing once).
* **Every refusal has a rule tag; no input may reach a panic.** Rewards come from untrusted generated programs: they run in a sandbox with a time and memory limit, and a crash of the program is a reward of zero, never a crash of the trainer.
* **No source file over 2,000 lines**; split by concern.
* **One GPU job at a time.** Check for other jobs (`pgrep`, `top`) before a run; stop a run that swaps.
* **Pin cancho-gpu by commit** (as cancho-gpu pins the compiler), and say in every result which commit of cancho-gpu produced it.
* **Implement in cancho where cancho can do it**, and use Python only where cancho-gpu's own examples do (driving sessions, plotting, comparing with torch). **[open]**: the split of the loop's code between the two is decided in R2 once the session driver's interface has been read.

## 9. Open questions for the maintainer

1. Model for R1: the 1.5B-Instruct (M8's choice) or the 0.5B first?
2. Task source: all of cancho-code's eval tasks, or a subset with deterministic checkers only?
3. Does R1 wait for M8's final result (the supervised baseline), or start from the base model now? (The M8 run is the first thing to finish; R0 and R1 do not depend on it.)
4. Is a negative R1 result acceptable as the end of the project? (The design says yes: section 4, R1.)

## 10. Limits of this document

Written from the documents and file names of cancho-gpu, not from running anything. The kernel contracts (the weighted loss, the adapters-off forward, the cost of merging adapters per iteration) and cancho-code's reward API are to be read, not assumed, before R0 and R2. All timing figures quoted are cancho-gpu's own, on one M4 Max; none was measured for RL.
