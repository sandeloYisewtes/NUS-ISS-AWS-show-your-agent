# 数据计划：从原型验证到可训练的 V7 Agent

## 一句话结论

没有公开数据集能完整提供 V7 所需的「屋主多轮证据 → Agent 干预 → 设计师方案 → 最终接受/返工」全链条。因此公开数据只能训练或验证**部件**；完整模型效果必须以已获同意的试点交互日志验证。

## 数据源分层

| 层级 | 可用数据 | 作用 | 不能声称 |
|---|---|---|---|
| 演示 | 本目录 `demo_events.jsonl`、状态空间模拟 | 验证接口、贝叶斯更新、硬约束和动作逻辑 | 真实用户满意度或商业效果 |
| 公开部件数据 | [Amazon ESCI](https://github.com/amazon-science/esci-data) | 查询—候选项相关性、检索/排序基线 | 装修偏好、设计师协作或 RL 效果 |
| 公开家居语料 | [Amazon Reviews 2023 / Home & Kitchen](https://amazon-reviews-2023.github.io/main.html) | 家具/材料属性、评论文本、弱监督偏好词典 | 原始装修搜索与方案接受 |
| 空间/家具候选库 | [3D-FRONT](https://arxiv.org/abs/2011.09127) 与 [3D-FUTURE](https://arxiv.org/abs/2009.09633) | 室内布局、家具对象、案例检索和方案特征 | 屋主动态偏好序列；下载前须同意数据许可 |
| RL/反事实工具 | [Open Bandit Pipeline](https://zr-obp.readthedocs.io/en/latest/) / [RecSim](https://github.com/google-research/recsim) | 动作日志、OPE 和模拟管线原型 | 真实装修业务效果 |
| 真实验证 | 已同意的屋主—设计师试点日志 | 验证 V7 端到端模型与后续 bandit/RL | 无 |

不要把 Amazon 公开研究数据误解为“可调用的 Amazon 全量用户浏览历史”；它们是固定发布的研究数据。现有购物比赛仓库可复用工程思想，但其服装数据不能成为装修模型的真实测试集。

## 第一批真实试点：最低日志要求

从第一天就让 Agent 保存下面四张逻辑表，且对用户做清晰授权、去标识化与数据保留说明：

```text
project
  project_id, pseudonymous_owner_id, area_sqm, room_type, budget_range,
  household_change, renovation_stage

evidence_event
  project_id, session_id, timestamp, x_text, image/case_id, a_it, g_it, c_it

agent_decision
  session_id, state_snapshot, candidate_set, u_it^Ag, shown_content,
  policy_version, propensity_mu_action_given_state

outcome
  session_id, next_feedback, explicit_confirmation, selected_case,
  plan_id, accepted, revision_count, satisfaction, hard_constraint_violation
```

`a_it` 是屋主已经发生的收藏/停留/选择等行为；`u_it^Ag` 才是 Agent 后续采取的 Ask、Show、Confirm、Recommend、Present、Defer。二者不能混写。`propensity_mu_action_given_state` 记录当时行为策略选择该动作的概率，是以后做 IPS/DR 反事实评估的必要字段。

## 严格训练、验证、测试切分

使用全局时间窗口，且以完整项目/完整会话轨迹为原子单位：

```text
Train:       t ≤ T0
Validation:  T0 < t ≤ T1
Test:        T1 < t ≤ T2
```

规则：

1. 同一 `project_id` / `homeowner_id` 的会话和方案版本只能在一个集合。
2. 在边界留 purge gap，至少覆盖最长会话和奖励归因延迟；跨边界的会话整段归入一个集合或删除。
3. 所有 TF-IDF、embedding、聚类、群体先验、归一化和缺失值处理仅在训练集拟合；验证集只选模型和阈值；最终测试集封存。
4. 除主时间测试外，留出“新屋主/新项目”冷启动测试，以检验群体先验。
5. 测试时的每个状态只可使用该时刻之前的信息，不能使用最终确认、最终接受、未来热门度等未来变量。

逐条随机切分或“每用户最后一条”会造成时间/项目泄漏。可参考 [Ji et al., 2020](https://arxiv.org/abs/2010.11060) 对推荐系统离线泄漏的讨论。

## GAN、模拟器与强化学习的使用边界

```text
真实训练集 → 拟合 GAN / 条件 VAE / 模拟器 → 只扩充训练集
真实验证集 → 调阈值、校准、模型选择
真实测试集 → 只做一次最终报告
```

- GAN / CTGAN / 条件 VAE 不能生成验证或测试数据，也不能替代真实满意度。
- 模拟器能测试“密度、质量、偏好转移和策略是否按逻辑工作”，但有 sim-to-real gap。
- 若只有自然浏览/搜索数据，可以训练 `p(feedback_next | state)`；没有 Agent 动作、候选集合、propensity 和后续结果时，不能可信地训练“哪种干预最好”的 RL。
- 先上安全、可解释的规则型策略；积累干预日志后做 constrained contextual bandit；最后才考虑多轮 RL。

## 现实可执行的三周节奏

| 周期 | 交付物 | 成功标准 |
|---|---|---|
| 第 1 周 | 200–500 张已授权案例卡、风格/材质/预算/户型标签；Agent MVP | 三轮案例选择可跑通，硬约束绝不违规 |
| 第 2 周 | 20–40 名试点屋主或模拟受试者的会话日志 | 可得到显式确认、案例选择和方案评分 |
| 第 3 周 | 冻结 split、基线评估、演示看板 | 对偏好确认、方案执行度和下一步动作分别报告指标与局限 |

最终展示时要清楚区分：**MVP 逻辑可执行** 与 **真实业务效果已经验证**。前者现在已具备，后者要靠严格试点数据完成。
