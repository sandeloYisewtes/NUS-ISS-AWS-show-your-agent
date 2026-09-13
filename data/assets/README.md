# 模拟素材库

本目录的三张图片均为 `synthetic_demo`：为本项目生成的 AI 模拟装修案例，不代表真实楼盘、真实屋主、真实设计项目或真实训练标签。

- [素材库表（CSV）](asset_catalog.csv)：便于人工查看、导入表格软件或制作演示页。
- [素材库表（JSON）](asset_catalog.json)：便于程序读取元数据、来源与使用限制。
- [标签规范](tag_spec.json)：定义可传入 Agent 的标签及其映射关系。
- `images/`：三张配套模拟案例截图。

## 可传给 Agent 的内容

在 MVP 中，图片不直接作为视觉模型的输入；调用 API 时只传已审核的 `agent_tags`、屋主文本和交互行为。例如：

```json
{
  "evidence": {
    "text": "想要温暖原木和足够收纳",
    "images": [{"asset_id": "asset_001", "tags": ["温暖原木", "收纳", "暖光"]}]
  }
}
```

`display_tags`、生成提示词、来源说明和隐藏评估标签不能被当作模型证据输入。
