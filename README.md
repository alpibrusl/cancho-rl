# cancho-rl

Reinforcement learning on [cancho-gpu](https://github.com/alpibrusl/cancho-gpu): rollouts, rewards, advantage estimators, the policy-gradient step and the experiments that measure them. The kernels, sessions and runners stay in cancho-gpu; the environments stay in `cancho-code` and `cancho-robot`.

**Status: R0 started.** A grader (`rl grade`) and 34 of the 200 small tasks exist, all in cancho; Python is only a comparison helper (`scripts/`). The rest is a design. Read [`docs/design.md`](docs/design.md) first: what is already in cancho-gpu, what is missing (rollouts), the milestones R0 to R4 and their exit criteria, and the experiment as it will be run.
