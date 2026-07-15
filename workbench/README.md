# 科研工作台

该目录是 FastAPI 的薄 Streamlit 客户端。它不导入领域模型、不加载模型制品，也不计算
SOH、RUL、预测区间或批次决策；界面只提交服务端 ID，并原样展示服务端签发的工作流
结果和审计 Markdown。

在 API 服务启动后运行：

```powershell
.\.venv\python.exe -m streamlit run workbench\streamlit_app.py
```

若环境未安装可选依赖，安装项目的 `streamlit` extra 后再启动。工作台需要 API 暴露：

- `GET /health`
- `GET /v1/tools`
- `POST /v1/batches/canonical-csv`
- `POST /v1/workflows/lifetime-decision`
- `GET /v1/reports/{result_id}`
