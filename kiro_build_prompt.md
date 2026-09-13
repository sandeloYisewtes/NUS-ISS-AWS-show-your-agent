# 给 Kiro 的首条开发任务

```text
You are working in a GitHub repository that already contains a shopping challenge agent.
Do not modify its existing agent.py, evaluator, or public evaluation files.

Create a new isolated folder named design_preference_agent for a V7 renovation
preference consensus Agent. Work on a new branch, add unit tests, and create a
pull request only after tests pass.

Product goal:
Help a homeowner and designer converge on an interior-renovation direction.

Required causal boundaries:
1. D_it = raw evidence: text, image tags, homeowner's completed actions, quality features.
2. c_it = true external context: budget, floor plan/area, family change, project stage.
3. z_it = hidden, dynamic homeowner preference. It must be inferred, never directly overwritten.
4. A_it = anomaly density. It only lowers q_it and increases observation noise R_it.
   It must not directly change z_it and must not be reused as an agent action or reward.
5. a_it = homeowner's past actions. u_it^Ag = the agent's next action. Keep them separate.
6. Q_z(delta_t) is state process noise. Do not reuse Q for an RL value function.

Implement a deterministic, explainable MVP first:
D_it + c_it -> density/quality -> Bayesian belief -> plan alignment E_ij and hard constraints G_ij -> safe next action.

Actions: Ask, Show, Confirm, Recommend, Present, Defer.
Hard constraints always filter actions before scoring. Use a rule-guided contextual-bandit baseline, not an unvalidated full RL implementation.

Create:
- a small Python API with reset/observe/evaluate_plans/decide/respond;
- scripted 3-turn Chinese demo data;
- tests for posterior update, budget constraint safety, and no project leakage in data splits;
- a README that clearly distinguishes synthetic-demo logic from real-world validated performance;
- data schema that records project ID, timestamps, evidence, context, plan outcomes, and intervention logging required for later bandit/RL training.

Use only the standard library unless a dependency is justified. Do not claim that Amazon clothing data validates interior-design preferences. Run the tests and report the results in the PR.
```
