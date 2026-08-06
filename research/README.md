# 电芯寿命预测顶级模型研究镜像

本目录用于论文阅读、官方实现审计、复现实验和泉芯智寿模型升级。`catalog.json` 是唯一来源目录；实际仓库、PDF和归档保存在被主Git忽略的本地子目录。

## 同步

```powershell
.\.venv\Scripts\python.exe scripts\research\sync_battery_research.py --continue-on-error
```

同步器对GitHub项目执行完整克隆，不使用浅克隆、对象过滤或单分支选项。再次运行会抓取所有远端更新、分支和标签，并执行 `git fsck --full`。

## 学习顺序

1. PBT、CyclePatch-BatLiNet、DITING：早期循环寿命主模型候选与挑战模型。
2. BatteryMFormer、HybridPatch、BatteryGPT：SOH退化轨迹、拐点和EOL研究。
3. BLAST-Lite、Naumann：循环—日历工况情景推演及大型方形LFP参考。
4. Smart Feature Identification：40Ah/280Ah短充电片段SOH特征。
5. BattGP：160Ah现场LFP系统在线健康监测与异常分析。
6. MAGNet：跨工况、跨域退化轨迹泛化。
7. IC2ML、iMOE、PINN4SOH、DiffBatt、SambaMixer：多任务、梯次利用、物理约束及概率轨迹研究储备。

## 安全规则

- 上游代码默认只读研究，未经审计不进入项目运行环境。
- 不加载上游未知pickle、joblib、`.pt`、`.pth`或预训练权重。
- 不沿用按行随机划分、全寿命百分比截断或测试集参与调参的实验设置。
- 复现时统一转换为项目的电芯级划分、固定20/50/100/150截断和可追溯制品。

## 当前特殊入口

- DITING已公开官方GitHub仓库，但代码仓库较新；复现前仍需独立核验许可证、数据划分和运行依赖。
- BatLiNet官方代码位于CodeOcean。匿名访问受限时，需要登录后手动执行Capsule完整导出，保存到 `research/archives/`。
- Hugging Face目前主要用于BatteryLife处理数据和论文索引，不采用缺少论文证据的个人模型权重。
