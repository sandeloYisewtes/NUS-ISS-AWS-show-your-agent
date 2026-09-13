# V7 装修偏好共识 Agent：可运行 MVP

这是一个可演示、可解释的第一版智能体，不是拿没有测试集的数据训练出来的“完整强化学习模型”。它把 V7 文档中的核心链路跑通：

```text
会话级证据 D_it + 外生情境 c_it
  → 密度/注意力/质量 q_it
  → 贝叶斯偏好后验 Bel_it(z)
  → 方案执行度 E_ij + 硬约束 G_ij
  → 下一步安全动作（Ask / Show / Confirm / Recommend / Present / Defer）
```

## 先回答：现在应先做什么？

应当同时推进，而不是二选一：

1. **现在构建 MVP。** 用规则化、可解释的贝叶斯更新和受约束策略跑通闭环，方便比赛演示和发现产品缺口。
2. **同时冻结数据契约与测试协议。** 不训练无标签的完整 RL；先收集真实“系统动作 → 屋主反馈 → 最终接受/返工”的日志。
3. **有真实数据后再训练局部模块。** 先训练文本/图片编码、证据质量、反馈预测；有干预日志后再升级为 contextual bandit 或 RL。

“训练集很多”本身不造成过拟合。真正的风险是同一屋主/项目泄漏到测试集、用测试集调参、GAN 污染验证/测试，或根本没有独立测试集。

## 运行

核心 Agent 只需 Python 3.10+，不依赖第三方包；可视化 Demo 额外使用 Streamlit。

```powershell
cd 'D:\NUS-ISS AWS show me your agent\design-preference-agent'
python demo.py
python -m unittest discover -s tests -v
```

`demo.py` 会回放三轮同一屋主会话，并打印：证据质量、注意力权重、偏好后验、两个设计方案的 `E_ij/G_ij`，以及 Agent 的下一步动作。

## 可视化现场演示

`app.py` 是面向评委和设计师的 Streamlit 可视化 MVP：输入屋主文本、图片标签、收藏/停留等交互信号、预算/面积/结构约束，以及候选方案，即可展示偏好后验、证据质量与异常密度、方案执行度和 Agent 的下一步建议。

```powershell
cd 'D:\NUS-ISS AWS show me your agent\design-preference-agent'
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器会自动打开本地演示页。`data/demo_plans.json` 中的方案和页面的预填内容仅用于展示模型流程，**不是**真实训练数据，也不代表真实业务效果。

比赛现场请直接使用侧栏的“比赛演示预设”；完整操作顺序、3 分钟讲稿与评委追问速答见 [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md)。预设内容位于 [data/demo_scenarios.json](data/demo_scenarios.json)，其中第 1 至第 3 轮必须使用同一项目 ID 连续运行，才能展示动态后验更新。

## 独立 Agent API

除了网页，`agent_api.py` 把同一个 `DesignPreferenceAgent` 封装为可由任意前端或工作流调用的本地 HTTP 服务。它保留项目会话状态，并把每一次重置和推理追加写入 gitignored 的 `.runtime/agent_audit.jsonl`，方便比赛演示追溯；该 MVP 的状态仍在内存中，因此**重启服务会清空当前项目状态**，审计日志不等同于训练数据或生产数据库。

```powershell
cd 'D:\NUS-ISS AWS show me your agent\design-preference-agent'
python -m pip install -r requirements.txt
uvicorn agent_api:app --reload --port 8000
```

打开 `http://127.0.0.1:8000/docs` 可直接试用带校验的 OpenAPI 页面。主要接口为：

- `GET /health`：服务和内存项目数；
- `POST /projects/{project_id}/reset`：创建或重置一个项目，可给入已确认偏好先验；
- `POST /projects/{project_id}/turn`：提交一个会话级证据包，获得偏好后验、方案对齐和下一步 Agent 动作；
- `GET /projects/{project_id}/state`、`GET /projects/{project_id}/history`：读取当前状态和本进程内会话历史。

接口契约测试覆盖健康检查、重置、一次完整推理、状态/历史读取、JSONL 审计和非法输入的 `422` 拒绝。

## 这版实现了什么、没有声称什么

| 已实现 | 当前未实现（需真实数据） |
|---|---|
| 会话级文本/图片标签/交互证据 | 用大规模真实装修文本训练的注意力编码器 |
| 异常密度 → 质量分 → 观测噪声 | 经过校准的异常检测阈值 |
| 一阶马尔可夫贝叶斯式状态更新 | 已验证的真实偏好因果模型 |
| 方案执行度 `E_ij`、硬约束 `G_ij` | 真实屋主满意度/返工预测效果 |
| 安全规则型动作策略 | 经离线反事实评估或线上随机实验训练的 RL |

因此比赛中应说：**“我们已交付可解释 Agent MVP 与可验证的数据/评估框架；模型参数会在严格隔离的真实数据上学习。”** 不应说已证明真实业务效果。

## V7 变量边界

| 变量 | 本项目含义 |
|---|---|
| `D_it` | 文本、图片、屋主已发生行为、质量特征组成的证据包 |
| `c_it` | 预算、面积、家庭变化、装修阶段等外生情境 |
| `z_it` | 不可直接观察的真实偏好；以逐维后验均值/方差近似 |
| `q_it` | 当前证据质量；仅通过观测噪声 `R_it` 影响本次更新 |
| `A_it` | 异常密度；不直接修改 `z_it`，也不是 Agent 动作 |
| `Q_z(Δt)` | 偏好自然变化的过程噪声；与 `q_it/R_it` 分离 |
| `a_it` | 屋主已经发生的收藏、停留、选择、跳过等行为 |
| `u_it^Ag` | Agent 的下一步辅助动作；与屋主行为分开 |

## 数据源与数据契约

Amazon/服装购物数据可以作为现有 GitHub 仓库的工程参考，但**不能用来验证装修偏好模型**。真实数据最小单元见 [data/schema.json](data/schema.json)：一条会话必须包含项目/屋主 ID、时间、证据、外生情境、候选方案与最终结果。公开数据、试点日志和严格切分的完整选择请见 [DATA_PLAN.md](DATA_PLAN.md)，离线 RL 与防泄漏依据见 [docs/research/sequential_offline_rl_protocol.md](docs/research/sequential_offline_rl_protocol.md)。

可先并行寻找四类数据：

- **公开图片/案例数据**：作为风格标签和检索语料；用于原型，不等于真实偏好真值。
- **装修平台/设计机构合作日志**：搜索、收藏、方案版本、确认、接受、返工；这是最终验证的关键。
- **受试者任务数据**：让小规模真实用户完成 3 轮案例选择与方案评价，获得第一批干净标签。
- **状态空间模拟数据**：人工设定隐藏的 `z_it`，检验贝叶斯恢复与策略逻辑；不能替代真实满意度。

## 训练、验证、测试：必须先写死的规则

1. 以 `project_id` 或 `homeowner_id` 为分组单位；同一项目的所有会话、方案版本只能进入一个 split。
2. 按项目最早时间排序，建议 70% / 15% / 15% 为训练 / 验证 / 测试；测试集来自最后一段时间的新项目并冻结。
3. 只在训练集拟合参数；只在验证集选阈值、奖励权重、模型版本；测试集只在最终一次使用。
4. GAN/CTGAN/条件 VAE 只可在训练折中拟合和扩充训练折，绝不生成验证或测试样本。
5. 若没有日志说明系统展示/追问了什么、可选候选集合和展示概率，则只能训练**下一步行为预测**，不能可信地训练“什么干预最好”的 RL。

[`evaluate.py`](evaluate.py) 提供了最小的按项目和时间切分函数，以及泄漏断言。

## 与现有 GitHub 仓库的关系

`sandeloYisewtes/sandelo-nus-techjam-shopping-AI` 的“状态维护 → 可解释决策 → 评估隔离”工程模式值得复用；但其 Amazon 服装商品、类别和评测结果不能移植为装修模型的数据或效果证据。本目录故意独立，避免破坏原 `agent.py`。

## Kiro 使用方式

Kiro Web 应作为开发协作者，而不是生产运行时。它目前要求 Pro 或更高计划；Free 账户不能使用 `app.kiro.dev` 的网页开发界面。可用时，连接 GitHub 仓库、创建独立分支，并将 [kiro_build_prompt.md](kiro_build_prompt.md) 作为第一条任务粘贴给它。要求它只在新目录内继续迭代并运行测试，不改原购物 Agent。参见 [Kiro Web 设置说明](https://kiro.dev/docs/web/setup/) 和 [GitHub 连接说明](https://kiro.dev/docs/web/github/)。
