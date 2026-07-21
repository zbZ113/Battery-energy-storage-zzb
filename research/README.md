# 电芯寿命预测顶级模型研究镜像

本目录用于论文阅读、官方实现审计、复现实验和泉芯智寿模型升级。`catalog.json` 是唯一来源目录；实际仓库、PDF和归档保存在被主Git忽略的本地子目录。

## 同步

```powershell
.\.venv\Scripts\python.exe scripts\research\sync_battery_research.py --continue-on-error
```

同步器对GitHub项目执行完整克隆，不使用浅克隆、对象过滤或单分支选项。再次运行会抓取所有远端更新、分支和标签，并执行 `git fsck --full`。

## 学习顺序

1. DITING：早期微弱退化信号与健康原型。
2. CyclePatch：循环级Token化，优先改造现有CPMLP。
3. BatLiNet：电芯内/电芯间联合寿命学习。
4. BatteryGPT：未来充电特征生成、SOH轨迹、拐点和EOL。
5. BatteryMFormer：时序/SOC双视角和退化模式记忆。
6. IC2ML：SOH、轨迹和RUL统一多任务学习。
7. iMOE：面向梯次利用的可解释专家混合。
8. PINN4SOH：物理约束与跨数据集稳定预测。
9. DiffBatt：概率退化轨迹和数据增强。
10. SambaMixer：Mamba长序列SOH编码。

## 安全规则

- 上游代码默认只读研究，未经审计不进入项目运行环境。
- 不加载上游未知pickle、joblib、`.pt`、`.pth`或预训练权重。
- 不沿用按行随机划分、全寿命百分比截断或测试集参与调参的实验设置。
- 复现时统一转换为项目的电芯级划分、固定20/50/100/150截断和可追溯制品。

## 当前特殊入口

- DITING尚未公开官方代码，目录只登记论文。
- BatLiNet官方代码位于CodeOcean。匿名访问受限时，需要登录后手动执行Capsule完整导出，保存到 `research/archives/`。
- Hugging Face目前主要用于BatteryLife处理数据和论文索引，不采用缺少论文证据的个人模型权重。
