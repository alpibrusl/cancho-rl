# cancho-rl

Reinforcement learning on [cancho-gpu](https://github.com/alpibrusl/cancho-gpu), for language models first and small control policies for cancho-robot second: rollouts, rewards, advantage estimators, the policy-gradient step and the experiments that measure them. The kernels, sessions and runners stay in cancho-gpu; the environments stay in `cancho-code` and `cancho-robot`.

**Status: a design, nothing built.** Read [`docs/design.md`](docs/design.md) first: what is already in cancho-gpu, what is missing (rollouts), the milestones R0 to R4 and their exit criteria, and the experiment as it will be run.
