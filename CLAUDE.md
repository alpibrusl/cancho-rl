# CLAUDE.md — cancho-rl

Read [`docs/design.md`](docs/design.md) before writing anything: it fixes the first target (GRPO with LoRA on a small Qwen, rewards from cancho-code tasks), the milestones R0 to R4 with their exit criteria, and the working rules (section 8). The ones that matter most:

- **Design before code**, in `docs/`, claims marked [read] / [derived] / [estimate] / [unknown]; a false claim is corrected in place, dated.
- **The gate passes before anything is called done**, and every check has been watched failing once.
- **Rewards run untrusted generated programs**: sandboxed, time- and memory-limited; a crash of the program is reward zero, never a crash of the trainer.
- **One GPU job at a time**; check `pgrep`/`top` first; stop a run that swaps.
- **Pin cancho-gpu by commit** and say in every result which commit produced it.
- **No source file over 2,000 lines.**
- **Pure cancho.** The package builds, grades and trains in cancho; Python may only compare with other frameworks (`scripts/`). `scripts/gate.sh` is the gate.
