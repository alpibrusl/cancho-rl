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
