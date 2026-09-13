# 单个屋主案例填写模板

> 用途：收集一位屋主在一个装修项目中的多轮偏好证据。一个 `project_id` 的所有会话、方案版本和最终结果必须始终在同一数据切分中。真实项目不得填写姓名、电话、精确地址等个人信息。

## 1. 案例基本信息

| 字段 | 填写内容 |
|---|---|
| `case_id` | `case_...` |
| `project_id` | `project_...` |
| 数据切分 | `development` / `validation` / `test` |
| 数据来源 | `synthetic_demo` / `owner_consent` / `licensed` |
| 是否合成演示 | `true` / `false` |
| 素材同意／许可引用 | 例如 `consent_...` 或 `license_...` |
| 是否包含个人可识别信息 | 必须为 `false` |

## 2. 户型与不可变约束 `context`

| 字段 | 示例 |
|---|---|
| `budget_max` | `180000` |
| `area_sqm` | `82` |
| `family_members` | `3` |
| `rooms` | `3` |
| `stage` | `inspiration` / `preference_refinement` / `brief_confirmation` |
| `no_structural_change` | `true` |

## 3. 会话级证据包 `D_it`

每一行是一场浏览／决策会话，不是一条点击。连续点击可按 30 分钟无操作间隔或项目阶段划分为一场会话。

| `session_id` | `timestamp` | 屋主文本 `evidence.text` | 素材 ID 与审核标签 `evidence.images` | 行为 `actions` | 质量 `quality` |
|---|---|---|---|---|---|
| `s01` | ISO-8601 时间 | “喜欢温暖原木，希望有收纳” | `asset_001`: `温暖原木, 收纳, 暖光` | `save`, `dwell_seconds: 60` | `source_reliability: .9`, `duplicate_ratio: 0`, `event_density: 1.1` |
| `s02` | ISO-8601 时间 | … | … | … | … |
| `s03` | ISO-8601 时间 | … | … | … | … |

只可把 [标签规范](../assets/tag_spec.json) 中的 `agent_tags` 传给当前 MVP。素材路径、生成提示词、来源、版权和展示标签不进入 Agent 的 `evidence`。

## 4. 候选方案

| `plan_id` | 名称 | `estimated_cost` | `min_area_sqm` | `requires_structural_change` | 七维 `features` |
|---|---|---:|---:|---|---|
| `plan_a` | … | … | … | `false` | `new_chinese`, `modern_minimal`, `warm_wood`, `warm_lighting`, `storage_priority`, `open_space`, `budget_sensitive` 均为 0–1 |
| `plan_b` | … | … | … | … | … |

## 5. 隐藏评估标签（必须与 Agent 输入隔离）

此部分只在所有会话推断完成后读取，不能放入 `evidence`、`baseline` 或候选方案特征中。

| 字段 | 填写内容 |
|---|---|
| `confirmed_preferences_after_all_sessions` | 人工预设或屋主最终确认的偏好 |
| `preferred_plan_id` | 屋主最终选中的方案 |
| `revision_count` | 修改轮数 |
| `satisfaction_score_1_to_5` | 满意度（真实项目需取得同意） |
| `known_model_gap` | 现有模型未覆盖的细粒度偏好，例如“克制 vs. 繁复” |

评估输出不一致应记录为模型边界，而不是把答案回灌到同一次推断中。
