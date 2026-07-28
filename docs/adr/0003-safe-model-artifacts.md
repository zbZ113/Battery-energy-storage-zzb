# ADR-0003：模型只使用可验证的非执行制品

- 状态：Accepted
- 日期：2026-07-28

## 背景

pickle、joblib、`.pt` 和 `.pth` 等格式可能在反序列化时执行代码，也可能掩盖模型结构、
上下文和依赖错配。模型注册若只保存文件路径，也无法证明线上字节与正式训练产物一致。

## 决策

1. 神经网络权重优先使用 safetensors；
2. 元数据、模型结构、特征、任务和运行上下文使用 JSON；
3. 表格数据使用 Parquet，接口交换使用受校验 JSON；
4. 每个 artifact 必须绑定文件大小、SHA-256、manifest 和 source revision；
5. Advanced artifact v2 绑定具体 candidate、seed、best epoch、run 和 checkpoint；
6. 路径必须位于受管根目录内，拒绝路径逃逸和符号链接；
7. 禁止未经来源核验加载 pickle、joblib、`.pt`、`.pth`；
8. 浏览器不能提交模型路径或哈希，服务端通过 artifact identity 解析。

## 备选方案

- 直接加载训练目录中的任意 checkpoint：不可审计，拒绝；
- 只校验扩展名：不能防止内容替换，拒绝；
- 把模型字节写入 PostgreSQL：增加数据库压力并模糊责任，拒绝；
- 只依赖镜像 tag：tag 可变，仍需 artifact 文件哈希。

## 后果

正面：

- 降低可执行反序列化风险；
- 可证明模型、清单和部署注册表一致；
- active route 可绑定具体而非聚合候选；
- 可以在离线 A100 与在线 ECS 间进行哈希交付。

代价：

- 每种模型需要明确、安全的恢复代码；
- artifact manifest 更新必须与文件原子交付；
- 大型结果和权重不能直接提交 Git。

## 实现证据

- `src/quanxin_life/application/advanced_deployment_registry.py`；
- `src/quanxin_life/application/model_artifact_catalog.py`；
- `src/quanxin_life/application/advanced_runtime.py`；
- `src/quanxin_life/training/` 中的安全 checkpoint 和输出索引；
- Advanced Final 结果中的 safetensors 与 SHA-256 清单；
- artifact、runtime resolver 和安全恢复相关测试。

## 成熟度

- Advanced Final artifact 与离线验证：Validated；
- 15 个受管候选注册代码：Validated；
- ECS 受管目录真实激活：尚未 Deployed。
