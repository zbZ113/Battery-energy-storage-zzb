# 科研工作台

该目录是 FastAPI 的薄 Streamlit 客户端。它不导入领域模型、不加载模型制品，也不计算
SOH、RUL、预测区间或批次决策；界面只提交服务端 ID，并原样展示服务端签发的工作流
结果和审计 Markdown。

## 导航区块

- **服务与数据**：检查 API、发现工具，并将审核后的 Canonical CSV 注册到服务端。
- **工作流执行**：只提交 `record_batch_id`、`calibration_cohort_id` 和 `policy_id`。
- **结果展示**：按工作流返回的结果 ID 读取数据质量、寿命点预测与区间、批次决策。
- **审计报告**：查看 ToolResult 的工具/模型/数据/特征版本、输入哈希、警告、来源记录，
  并原样展示和下载服务端生成的 Markdown。

展示层不会通过点预测计算 RUL，不会根据区间重新执行决策，也不会用默认值补齐缺失字段。
FastAPI 响应中没有对应字段时，界面明确显示“后端未提供”。零值与 `false` 会作为真实后端值
展示，不会被误判为缺失。

在 API 服务启动后运行：

```powershell
.\.venv\python.exe -m streamlit run workbench\streamlit_app.py
```

若环境未安装可选依赖，安装项目的 `streamlit` extra 后再启动。工作台需要 API 暴露：

- `GET /health`
- `GET /v1/tools`
- `POST /v1/batches/canonical-csv`
- `POST /v1/workflows/lifetime-decision`
- `GET /v1/results/{result_id}`
- `GET /v1/reports/{result_id}`
