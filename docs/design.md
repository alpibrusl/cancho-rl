# cancho-rl: reinforcement learning on cancho-gpu (a design, written before the code)

> Amended 2026-11-04 (second version): the scope is any policy with log-probabilities and a weighted gradient step, with two tracks (language models first, small control policies second); see sections 1 and 4b.
>
> Amended 2026-11-05 (third version, reviewing the second): the repository is a **generic RL framework whose first instance is the language-model track**. Section 4c names the seams that make it generic (policy, environment, rollout, estimator, objective, evaluator), says which of the code written so far is generic and which is LLM-only, and fixes when the shared layer is extracted (when the second instance needs it, not before). Section 1's "two tracks" paragraph is reworded to match.

> Written 2026-11-04, before any code. **Nothing in this document is built or measured in this repository.** Claims carry a mark: **[read]** from a file of another repository (named), **[derived]** from numbers that were measured elsewhere, **[estimate]** a guess with its reasoning, **[unknown]** not looked at yet and to be read before the milestone that depends on it. A claim that turns out false is corrected here, in place, with the date.

## 1. What this is, and what it is not

`cancho-rl` is where reinforcement-learning *algorithms* live, for **any policy that can give the log-probability of its actions and take a weighted gradient step**: a language model (the first track, sections 2 to 7) and a small control network for a robot (the second track, section 4b): the rollout loop, rewards, advantage estimators, the policy-gradient loss and the experiments that measure them. It runs on top of [`cancho-gpu`](https://github.com/alpibrusl/cancho-gpu), which owns the kernels, the training sessions and the runners.

The split, and why:

* **cancho-gpu keeps the primitives.** A kernel, a session directive, a faster decoder, a numerically checked loss gradient. Its gate holds numerics to 1e-4 against torch and its documents are about the training stack; algorithms that change every week do not belong in it.
* **cancho-rl keeps the algorithms and the experiments.** GRPO, REINFORCE with a leave-one-out baseline, PPO if ever; reward functions; the evaluation protocol.
* **The environments stay where they live.** Tasks with a checkable reward for a coding model come from `cancho-code` (its eval harness and journals); a robot policy would come from `cancho-robot`'s simulator. `cancho-rl` calls them; they do not depend on it.

Not goals: a gym replacement or a catalogue of every algorithm (an algorithm is added when a track needs it and has a measured baseline), multi-GPU or NVIDIA training (cancho-gpu's CUDA path has run only kernels so far, not a training session), RLHF with a learned reward model.

**A framework with one instance built first.** The LLM instance (GRPO, language-model policies) is built first because its reward is checkable and its baseline is measured (section 2); the control instance (section 4b) is the second, and the seams between what is shared and what is not are in section 4c. The shared layer is not written ahead of the second instance: with one instance every guess about the seams is a guess, so the first instance is written with the seams named and kept clean, and the extraction happens when the second one shows which guesses were wrong. The control instance shares the parts that do not depend on the policy: the advantage estimators, the loss as a weighted log-probability objective, the evaluation protocol with seeds and bootstrap intervals, and the rule that a reference implementation in torch comes first. What differs is the regime: a language-model iteration is dominated by generating completions, a control-policy iteration by simulator steps and by many small gradient steps. Nothing in the algorithm layer may assume tokens, a vocabulary or a prompt.

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

## 4b. The second track: small control policies for cancho-robot (an outline; milestones are fixed when the environment has been read)

**Why it is worth having.** The measured wins of cancho-gpu over torch on MPS are on small networks, where launch overhead and fusion decide: the 60 M-parameter GPT step 15% faster, the MLP and CNN steps within a few percent either way, a CNN 1.6 times slower **[read: cancho-gpu `docs/resident.md` sections 13 to 19]**. A control policy is a small network (an MLP, or a small CNN on camera frames) updated many times, so the regime is the one where cancho-gpu has something to show; whether it shows it for RL is not known.

**First algorithm: PPO** (clipped surrogate, generalised advantage estimation, a value head sharing the trunk or a separate small network). It is the default baseline for continuous control; SAC or others are added only if PPO plus the task leaves a question they answer. **[open]**

**The environment.** cancho-robot's simulator, which owns the dynamics and the reward. What `cancho-rl` needs from it is an interface of a few calls (reset with a seed, step with an action, observation and reward, a done flag, a way to run many environments in a batch). Whether `cancho-robot` offers that, in a form callable from a session driver at the speed an RL loop needs, is **[unknown]** and is the first thing to read for this track. If the simulator is slow, the milestones below measure that before they measure anything about learning.

**Milestones (outline, exit criteria to be fixed before building each):**

* **C0. The environment and its baseline.** A task in cancho-robot with a reward and a way to run it in batches; a random policy and a scripted policy scored. Exit: environment steps a second, the reward of both baselines with seeds, and a check that the reward cannot be earned by a degenerate behaviour (the control analogue of R0(c)).
* **C1. A torch PPO reference.** Torch on MPS (or CPU, whichever is faster for these sizes), the same environment. Exit: the return after N environment steps above the scripted baseline's, with two seeds at least, or the negative result recorded.
* **C2. The PPO update in a cancho-gpu session equals torch's.** One fixed batch of trajectories: the loss and every parameter gradient against C1's, then a short run whose return curve is within the spread of two torch seeds. This is the point where the small-network advantage, if any, is measured: seconds per update against torch's.
* **C3. The loop end to end, with the simulator in the picture.** Report the fraction of wall time spent in the simulator, the policy forward, and the update. If the simulator dominates, the honest conclusion is that the update's speed does not matter for this task, and that is the result.

The risks of section 7 apply with the changes the regime brings: reward hacking appears as the policy exploiting simulator quirks (sim-to-real caveats stay in cancho-robot's documents; no number from simulation here says anything about the real arm), and RL on small networks is noisy enough that fewer than two seeds is not an experiment.

## 4c. The framework: six seams, and what is generic today (2026-11-05)

An RL run is the same loop whatever the policy is: *sample actions from the policy in an environment, turn rewards into advantages, take a weighted gradient step, measure on held-out instances*. The framework is the loop plus six interfaces; an instance (LLM, control) supplies the first two and shares the rest.

| Seam | What it is | LLM instance (first) | Control instance (second, outline) |
|---|---|---|---|
| **Policy** | `act(observation, seed) -> action, logprob`; `logprob(observation, action)`; `step(weights)`; adapter or parameter state to save and restore | Qwen with LoRA on cancho-gpu; an action is a completion (tokens) | an MLP or small CNN; an action is a vector |
| **Environment** | a named, seeded instance: `reset(seed) -> observation`, `step(action) -> observation, reward, done`; a one-shot environment is the special case of one step | a task from `tasks/<id>`; one step, the completion is the program; the reward is the grader's | a cancho-robot scene, many steps |
| **Rollout** | runs a batch of policy-environment episodes, records `(observation, action, logprob, reward, done)` | a sampling script (`scripts/sample.sh`, run in section 17) | simulator batch |
| **Estimator** | rewards of a group or trajectory to advantages: leave-one-out group baseline (GRPO), GAE (PPO) | group-relative, per task | GAE with a value head |
| **Objective** | advantages and logprobs to a weighted log-probability loss, with clipping and the KL term | token-level, masked to the completion | per-step |
| **Evaluator** | held-out instances, seeds, interval | pass rate at k samples, bootstrap interval | return over seeds, bootstrap interval |

Nothing above the line between Policy/Environment and the rest may assume tokens, a vocabulary, a prompt, a sandbox, or that a reward lies in [0, 1].

**What the code written so far is, said plainly [read from this repository: PR #2 and PR #3, both merged, the second holding `rl split`, `rl prompt`, `prompts/primer.md`, `scripts/sample.sh` and design section 16]:**

* **Generic already, and worth extracting later:** the held-out split rule (FNV-1a over an instance name, `tools/rl/main.cho`: `fnv`, `is_heldout`; it needs only a name); the reward record's shape `{id, reward, solved, status}`; the rule that a failure of the thing under test is a reward and never a failure of the trainer; the rule that a gate probe (an input that must score a known value) exists for every reward.
* **LLM-only, and staying in the LLM instance:** `lib/rl/grade.cho` (composes a test module from a candidate *program* and an oracle, runs `cancho test` under a deadline: the sandbox exists because the policy's output is untrusted code, which a control policy's is not), `rl prompt`, `prompts/primer.md`, `tasks/`, `scripts/sample.sh`.
* **Shared helpers, not a design question:** `lib/rl/text.cho`.
* The reward in the LLM instance is `passed/of` in [0, 1] plus `solved`; a control instance's reward is an unbounded return, so the record's `reward` is a float with no range promised and `solved` is optional. This is the first guess that would have been wrong if the framework had been written from the LLM case alone.

**Layout, and when it changes.** Today: `lib/rl/{text,grade}.cho` and `tools/rl/main.cho`. The shared layer moves to `lib/rl/core/` and the LLM code to `lib/rl/llm/` **when the first control milestone (C0) starts**, as one change that moves files and nothing else, so the history shows what the second instance forced. Until then no file is renamed and no interface is declared in code that nothing calls. This answers open question 6 (section 9): one repository, one directory per instance, the shared layer extracted at the second instance.

**What would make this framework the wrong shape, stated now.** (a) If the policy interface needs the whole sampling loop inside cancho-gpu (as decoding on a fused kernel may require, section 6), `act` is a batch call into the session, not a per-observation call, and the Rollout seam is the session's. (b) If control rollouts need the simulator and the policy in one process to be fast enough, the Environment seam is a batch API (`step` over N environments), which the one-shot LLM environment satisfies trivially with N = the group size. (c) If the objective for the two instances differs in more than masking and the estimator, the Objective seam is two functions behind one name, and that is a finding, not a failure.

**Python and torch** stay what section 14 made them: in the control instance, C1's torch PPO reference is a comparison helper under `scripts/`, never a dependency of the package.

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
5. Is the control track (section 4b) wanted, and which cancho-robot task is the first one? It can start with C0 as soon as the simulator's batch interface has been read; it does not wait for the LLM track.
6. Do the two tracks share a repository layout from the start (one `algo/` layer, one directory per track), or does the second track begin as a copy and get merged when the shared layer is clear? This design assumes the first. **Answered in the design 2026-11-05 (section 4c):** one repository, one directory per instance, the shared layer extracted when the second instance starts, not before; the maintainer may still overrule.

## 10. Limits of this document

Written from the documents and file names of cancho-gpu, not from running anything. The kernel contracts (the weighted loss, the adapters-off forward, the cost of merging adapters per iteration) and cancho-code's reward API are to be read, not assumed, before R0 and R2. All timing figures quoted are cancho-gpu's own, on one M4 Max; none was measured for RL. The control track (section 4b) is an outline written without having read cancho-robot's simulator interface; its milestones are placeholders until that is read.

## 11. R0 harness probe (2026-11-04): what cancho-code's eval offers as a reward

Read and run on cancho-code at `6fcac18`, compiler `target/debug/cancho` of this repository (a **debug** build; release timings are **[unknown]**). Nothing was changed in cancho-code.

* **[read]** The scoring entry point exists and is a file-to-score call: `CANCHO=<compiler> python3 eval/audit_<item>.py <project.cho>` prints per-function-group `passed/of` and `TOTAL n/N`. There are 14 audits (`basics`, `tokenbucket`, `roman`, `base64`, `duration`, `editdistance`, `sortedmap`, `jsonptr`, `expr`, `glob`, `semver`, `ini`, `csv`, `multi`); `tasks.json` lists 13 tasks. Oracle sizes run from 25 (`sortedmap`) to 873 (`multi`) cases. The audit wraps the candidate (strips `main`, makes functions `pub`, adds test helpers) and runs `cancho test`.
* **[measured]** `roman/reference.cho` scored 439/439 in three runs (0 disagreements; R0(a) wants 20, so this is 3 of 20). The empty program (`edition 7;` only) scored 0/439 with the compiler's error per group. Wall time per score was 7.4 to 10.4 s with 1.3 s user CPU, so most of it is waiting (compile or process startup), **[unknown]** which.
* **[read]** Safety: the only limit is `timeout=300` on one `cancho test` call (`audit_basics.py`; `audit_roman.py` has none). There is **no memory limit and no sandbox**. This does not meet the working rule for untrusted programs; R0 needs a wrapper (time and memory limits, crash = reward zero), not a change to the audits.
* **[derived]** The partial score (`n/N`, per group) is a usable dense reward; pass/fail would be `n == N`.
* **Correction to section 3, row "Rewards"**: it is no longer **[unknown]**; the API is the audit script above.
* **The gap in R0(b) stands, in measured form:** 14 items, not 200 tasks. A per-function-group split gives somewhat more units (roman has 3 groups) but nowhere near 200, and the audit shapes are fixed by each item's plan, so a task is a whole library built from a plan, not a short completion. Getting to 200 needs new tasks (a generator, or small single-function items with their own oracles). That is a design decision, left to the maintainer.
* **Not probed:** the "prints the expected output" and "exits 0 early" hacking probes (R0(c)): the audits call the candidate's functions, so those probes need an item-level test, written with the wrapper. Held-out separation (R0(b)) is **[unknown]**: oracles live in the same repository as `reference/`.

## 12. R0 task set: small single-function tasks (decided 2026-11-04, before the code)

> **Corrected 2026-11-04 (section 14): the Python grader, generator and `prompt.json`/`cases.json` described here were replaced by cancho.** What still holds: the task shape, the independent oracle, the split idea (now FNV-1a, not SHA-256), the reward `passed/of` plus `solved`. What changed: `tasks/<id>/` holds `task.spec`, `oracle.cho` and `reference.cho`; the grader is `rl grade`.

The maintainer chose to write small tasks rather than relax R0(b) (section 11). The design:

* **A task** is one pure function over `int`/`bool` (effect row `[]`), the shape of cancho-code's `basics` item, so the existing way of testing (a generated `cancho test` file, one `test_cNNN` per case, grouped by function) carries over. It has: `prompt.json` (name, signature, description, a few visible examples), `cases.json` (hidden oracle cases, JSON lines of `{args, expect}`), `reference.cho`.
* **Oracle independence:** `cases.json` is computed by a Python function written separately from `reference.cho`, plus hand-computed anchors asserted in the generator (as `basics/oracle.py` does). The reference must score full on its own cases; a case set that its reference fails is a bug in one of the two.
* **Split:** a task is held out when `sha256(id)` mod 4 == 0 (a fixed rule, not a choice made after seeing results). Held-out `cases.json` is read only by `grade.py` when called with a held-out id and never copied into any prompt or training record.
* **Grader `grade.py <id> <candidate.cho>`:** builds the test file, runs `cancho test` in a child with a wall-time limit and `RLIMIT_CPU`; prints one JSON object `{id, passed, of, reward, status}` with `reward = passed/of` and `status` one of `ok`, `build_error`, `trap`, `timeout`. Any child failure is reward 0; the grader itself never raises on a candidate. **[unknown]** whether macOS enforces `RLIMIT_AS`/`RLIMIT_DATA` for the compiler's output; the memory limit is set and its effect is measured before it is claimed.
* **Count:** the target stays 200. The first commit builds the machinery and a first batch; the number reached is stated in the commit, not the target.
* **Sandbox note:** the candidate is a cancho program compiled by `cancho test` with no declared effects, so it cannot reach the filesystem or network by language rule **[read: cancho's effect rows, AGENTS.md]**; the limits cover only loops and memory.

## 13. R0 first batch: what was built and measured (2026-11-04)

> **Superseded by section 15 for the tool; the measurements stand** (they were reproduced with the cancho grader, section 15).

Built: `tools/grade.py` (the grader), `tools/gen_tasks.py` + `tools/tasks_a.py` (definitions), `tasks/<id>/` (generated), `probes/` (hacking and failure probes). Compiler: this repository's `target/debug/cancho`.

* **[measured] 34 tasks, not 200.** 22 train, 12 held out by the hash rule (35%, not exactly 25%: the rule is fixed, the share is what it gives). Every reference scores full on its own cases (34/34, first run). The remaining 166 tasks are not written; R0(b) is **not met**.
* **[measured] Determinism (R0a):** 20 repeated gradings each of `fib` (reference), `fib` (a mutant) and `triangle_kind` (reference): 60 gradings, 0 disagreements.
* **[measured] Probes on `fib`:** empty program 0/91 (`build_error`); garbage 0/91; `while true {}` 0/91 (`timeout`, wall limit); division by zero 0/91; a program with an early-exit `main` that returns 0: 1/91 (its `fib` is the constant 0, which is right for n = 0 only); a constant-0 `fib`: 1/91; an off-by-one mutant: 1/91. None solves the task (`solved` is false); **the dense reward `passed/of` gives a trivial program about 1% on this task, and a task whose oracle is dominated by one common value would give more.** Whether RL should use the dense reward or `solved` is a decision for R1, and the per-task floor of the constant programs should be reported with the pass rate.
* **[unknown] Memory limit:** `RLIMIT_AS` is set but a program that allocates heavily was not run; no claim is made that memory is bounded. A program that tries `main` effects was not tried; the "prints the expected output" probe is moot for graded functions (`main` is stripped before grading).
* **[unknown] Speed:** about 8 s a grading on the debug compiler, 4 in parallel; 34 references took 26 s. Release-build time and the cost of 8 samples x 200 tasks per iteration are not measured.
* **[open]** Base-model pass rate (R0b: 10 to 70% with 8 samples) needs a model and has not been run. The tasks are mostly easy integer functions, so a coding model may sit above 70%; that is the first thing to measure before writing the other 166.

## 14. The package is cancho; Python is a comparison helper (decided 2026-11-04, before the port)

The maintainer's rule: **cancho-rl is pure cancho; Python may only be a helper that compares with other frameworks, or not be used.** Sections 12 and 13 built the grader, generator and oracles in Python; that is **corrected here**: those files are a first version, to be replaced, not kept. The pattern is the sibling's: cancho-opt is cancho with `scripts/*.py` that only compare it with scipy.

* **In cancho (the package):** the task spec parser, the test-file composer, the grader (spawn `cancho test` under a deadline, parse its summary), the task oracles (`tasks/<id>/oracle.cho`, written in a different style from `reference.cho`: brute force where the reference is clever), the JSON answer, and later the rollout/advantage/loss code that is not a kernel. Binary `rl`, built with `cancho build --bin rl`.
* **Task format (no JSON in the data, so no Python to produce it):** `tasks/<id>/task.spec`, line oriented `key: value`: `name`, `ret` (int or bool), `desc`, `example` (a call and its value, hand-checked), `grid` (one line of ints per parameter; the cases are their product, every k-th taken above 250) and `case` (explicit tuples). `oracle.cho` defines `o_<name>`; the composed test file asserts `name(args) == o_name(args)` per case, so no expected values are stored.
* **Python's remaining role (`scripts/`):** `crosscheck.py` evaluates the same functions with Python's own `math`, on the cases `rl cases <task>` prints, and fails on a disagreement: an independent oracle in another framework, as `check_scipy.py` is for cancho-opt. It is not needed to build, grade or train.
* **Siblings to reuse, by commit pin (`cancho add`), not by copying and not by shelling out** **[read: alpibrusl/cancho-tools and cancho-stats, shallow clones, 2026-11-04]**: `cancho-tools` has a `contract` package (flag parsing with rule-tagged refusals, a JSON answer envelope, bounded `read_all`, atomic writes) and a `run` tool that starts a program with a deadline and an output cap, the shape this grader needs; `cancho-stats` has a percentile bootstrap (`lib/csvx/boot.cho`), which is the success rule's interval in section 5. Which of them cancho-rl takes as a dependency is decided when the code first needs it; this section does not claim they fit.
* **Split rule corrected:** the hash is FNV-1a over the task id, mod 4 == 0 held out (the SHA-256 of section 12 needed a library the first tool does not otherwise use). Nothing was trained on the old split.
* **Before the Python files are deleted** the cancho grader must reproduce the Python grader's results: all 34 references full, the probes of section 13 at the same scores. Until then they remain, marked as superseded.

## 15. The port to cancho, measured (2026-11-04)

Built (cancho, edition 9 for `exec_spawn_detached`): `lib/rl/text.cho` (file reads, integers, lines), `lib/rl/grade.cho` (compose the test file, run `cancho test`), `tools/rl/main.cho` (`rl grade CANCHO TASKDIR CANDIDATE SCRATCH`), `cancho.toml`, and per task `task.spec` (`name`, `ret`, `sig`, `desc`, `example`, `case` lines), `oracle.cho` (written in a different style from the reference: recursion and pair-wise divisor loops where the reference iterates), `reference.cho`. `scripts/gate.sh` is the gate.

* **[measured]** All 34 references are `solved` by the cancho grader; the six probes score exactly what the Python grader gave (empty 0/91, garbage 0/91, division by zero 0/91, early-exit `main` 1/91, constant 0 1/91, off-by-one mutant 1/91); `loop_forever` is `timeout` with reward 0 and leaves no process behind (the wall deadline kills the group).
* **[measured]** Cross-check: `scripts/crosscheck.py` grades, for each task, a table program holding Python's own answers for every case against the cancho oracle: 34 of 34 agree. Watched failing once: changing `o_popcount` to `n % 3` makes it report `DISAGREE popcount` (3 of 24), and restoring it agrees.
* **[measured]** Speed: about 0.9 s a grading (the Python grader took 8 s), 34 references in 8.5 s with 4 in parallel, on the debug compiler. The earlier figure was dominated by Python and `sh` start-up, not the compiler.
* **Correction:** `ulimit -t 20` did **not** stop a runaway program in 20 s on this macOS (44 s of CPU were used before the 60 s wall deadline fired). The wall deadline is the limit that works; the CPU limit is not claimed. The memory limit is still unmeasured.
* **Python left in the package:** `scripts/crosscheck.py` and `scripts/oracles_py.py` only (comparison helpers). No Python builds, grades or trains.
* **Not built:** `rl prompt` (print a task as a prompt for the policy), grid expansion in `task.spec` (cases are listed one per line, as the Python version generated them), the split rule in cancho (FNV-1a is decided, not coded), the other 166 tasks, and every later milestone. Dependencies on cancho-tools and cancho-stats are not taken yet.
* **A process slip, recorded:** an unnecessary `git rm --cached -r .` staged the deletion of three tracked files; `git reset` restored the index, nothing was committed.

## 16. `rl prompt`, `rl split` and the sampling script (2026-11-04)

* **[measured]** `rl split TASKDIR` (FNV-1a 32 bit over the task name, mod 4 == 0 held out) gives 22 train and 12 held out; the same rule in Python (a comparison, in the gate) names the same 12 tasks. The 32-bit form is used because int overflow is an error in cancho, so a 64-bit FNV product cannot be computed without reducing it.
* **[measured]** `rl prompt PRIMER TASKDIR` prints `prompts/primer.md`, the signature, the description, the examples and the reply instruction; the gate checks that no prompt contains a `case:` line.
* **[unverified]** `scripts/sample.sh MODEL K OUTDIR` (mlx_lm as the model runner, the code block cut out with awk, each sample graded) was written and **not run**: an M8 training job (`m8_train.py`) held the GPU, and the working rule is one GPU job at a time. Whether `mlx_lm.generate --prompt` applies the chat template, and what a 1.5B model's output looks like, are unknown until it runs. The base pass rate of R0(b) is therefore still unmeasured.
* **Expectation, stated before the run:** a base Qwen2.5 has never seen cancho, so with this primer the pass rate may be near 0 on the harder tasks. That is the "no signal" risk of section 7 and the reason to measure before writing 166 more tasks.

## 17. First base-model pass rate (2026-11-05)

* **Corrected expectation (section 16 said near 0):** Qwen2.5-1.5B-Instruct, primer v1, temperature 0.8, 600 tokens, 8 samples on each of the 22 train tasks (176 samples), graded by `rl grade`, compiler `8812bd5`, `mlx_lm.generate` on the M-series GPU (no other GPU job was running): **30 of 176 samples solved (17%)**; 48 built and ran, 128 (73%) did not build. `mlx_lm.generate --prompt` applies the chat template (the replies are one clean fenced block).
* **Per task [measured]:** 11 of 22 tasks have a solve rate in the R0(b) window (above 0 and at most 0.7: abs_diff, digit_sum, fib, fizzbuzz_code, floor_div, hamming_bits, is_prime, max3, mod_floor, pow_int, round_up_to); `is_even` is solved 8/8; 10 tasks are 0/8 (bit_length, count_digits, count_divisors, days_in_month, digit_sum_base, is_palindrome_num, is_perfect_square, popcount, sum_proper_divisors, sum_squares_to), eight of them with 8 build errors. So the signal exists (the "no signal" risk of section 7 did not materialise on this model) but it is thin: the median task is 0.
* **What the build errors are [measured]:** 63 of the 128 are ``expected an expression, found `=` `` (the model writes `x += 1`, `n /= 10`, which cancho does not have), 16 ``found `[` ``, 8 a missing `;`, 6 `?`, 4 assigning to an immutable parameter, and `abs` is not a function. The primer did not say any of this. That is a prompt defect, not a model one: the reward for these samples is 0 for a reason the model could be told.
* **Consequence:** primer v2 states the missing rules (no compound assignment, no `abs`/`min`/`max` builtins, parameters are immutable). The v1 numbers above stay as the baseline; the v2 run is reported below it, with the same seeds. Not yet shown: 8 samples is a wide interval per task (the cancho-stats bootstrap is not used yet), the held-out tasks were not sampled, and a single model.

**Primer v2 result [measured, same model, seeds and tasks]:** **28 of 176 solved (16%)** against v1's 30 (17%); 121 build errors against 128; ``found `=` `` fell from 63 to 47. That is inside the noise of 8 samples a task: **the one-line rule did not fix the habit**, a 1.5B model keeps writing `+=` after being told not to. Solve counts move task to task (sum_squares_to 0 to 4, round_up_to 3 to 0), which is what sampling noise looks like. **Conclusions:** (1) R0(b) holds as measured: about 16 to 17% of samples solve, 11 of 22 tasks sit in the 10 to 70% window, and 10 tasks are at 0; (2) the dominant failure is syntax the model brings from other languages, which is the thing a policy-gradient step can learn and a longer prompt did not; (3) the primer stays at v2 (it is not worse) but no more time goes into wording it; (4) the next measurements are the held-out tasks (12) and the 7B/0.5B for the size dependence, each a separate run. The sample directories are in `work/` (ignored by git); the results are the counts above.

## 18. Held-out base rate, and 13 more tasks (2026-11-05)

* **[measured] Held-out base rate:** Qwen2.5-1.5B-Instruct, primer v2, 8 samples on each of the 12 held-out tasks of the 34-task split (96 samples): **17 solved (18%)**; 59 build errors, 35 built and ran, 2 timeouts (a candidate that looped: the wall deadline fired, reward by status `timeout`). Per task: sign 7/8, triangle_kind 2, ceil_div 2, factorial 2, and 1 each on is_power_of_two, collatz_steps, reverse_digits, cube_root_floor; 0 on median3, digital_root, binomial, sum_to. That is the same rate as the train tasks (16 to 17%), so the split is not easier or harder by construction, which is the only thing this checks. 4 of 12 held-out tasks are at 0.
* **13 tasks added (47 in all):** gcd, lcm, clamp, is_leap_year, min3, sum_odd_to, is_odd, sum_range, digit_product, max_digit, sum_multiples, trailing_zeros_fact, abs_val. Each has an oracle written differently from its reference (closed form against loop, recursion against loop). The split moves with the task set (16 held out, 31 train); the gate used to compare against a fixed list and now compares against the same FNV-1a rule written independently in Python (a comparison helper), and the comparison was watched failing once with the rule broken. The 13 are not yet sampled; the 34-task numbers above are not changed by them. Still short of 200 by 153.
