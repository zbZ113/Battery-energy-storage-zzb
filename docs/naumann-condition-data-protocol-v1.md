# Naumann 工况级退化数据协议 v1

## 目的与边界

本协议接入 Naumann 公开的 LFP/石墨电芯循环老化与日历老化数据，服务于：

- 工况—退化率/容量损失关系建模；
- Naumann 高斯过程（GP）与主动试验回放；
- 温度、平均 SOC、DOD、充放电倍率的工况敏感性分析；
- 储能静置工况的日历老化参考。

这些数据**不等价于企业储能电芯数据**，也不是用于 `CycleRecord`、早期循环 `EOL80` 预测或 CPMLP 训练的电芯级寿命样本。适配器不得将条件级汇总测量伪装成逐采样点电化学曲线。

## 本地数据实况与登记

| 数据集 | 官方格式 | 本地格式 | 当前接入状态 | 合法用途 |
| --- | --- | --- | --- | --- |
| `NAUMANN_CYCLE` | MATLAB v5 `.mat` | 21 个 `.mat` | 源目录已登记为 `matlab`；MAT 条件适配器待实现 | GP、主动试验、工况退化率 |
| `NAUMANN_CALENDAR` | `.xlsx` | 4 个 `.xlsx` | 容量工作簿的显式布局适配器已实现 | 日历老化容量轨迹与温度/SOC工况分析 |

`NAUMANN_CYCLE` 文件包含容量、直流内阻、EIS、dV/dQ 及工况汇总矩阵。它们不应通过 Excel 通用适配器读取。

`NAUMANN_CALENDAR` 的容量、阻抗、EIS 与微分电压工作簿由不同指标组成，首版只接入 `DischargeCapacity.xlsx` 的容量数据；其余指标必须各自有明确单位与布局契约后再接入。

## Calendar 容量工作簿契约

`load_naumann_calendar_capacity` 接收：

1. 已验证 SHA-256、来源 URI、许可证的 `RawFileManifest`；
2. 与该来源完全匹配的 `SourceCatalogEntry`；
3. 由人工审阅的 `NaumannCalendarLayout`。

布局必须显式声明：

- 工作表名称；
- 容量标题所在单元格及其精确文本，确认单位为 Ah；
- 储存时间标题、列和首个观测行；
- 每个条件列的精确标题、条件 ID、温度和平均 SOC。

适配器不会从类似 `TP_40°C,50%SOC` 的字符串自动猜测物理条件。时间必须严格递增；任一条件值缺失、非有限、负容量、标题不符或来源不符都会阻断读取，而不是插值、补零或重排行。

输出 `CalendarCapacityObservation`，其字段包括储存时间、温度、平均 SOC、容量、源文件 SHA-256、布局版本和适配器版本。Calendar 数据没有循环 DOD 或 C-rate 时，三个字段保持显式 `null`。

## 版本、可追溯与 Git 规则

- 原始下载目录 `data/MATR/`、`data/HUST/`、`data/NAUMANN_CYCLE/` 和 `data/NAUMANN_CALENDAR/` 一律不提交 Git；
- 数据卡、来源 URI、许可证、SHA-256、转换配置与生成的不可执行 Parquet 清单才进入版本控制；
- HUST 的 `.pkl` 继续隔离，禁止本进程反序列化；
- `.mat` 与 `.xlsx` 仅在来源和 SHA-256 核验成功后读取；
- 对外报告应标记这些数据为公开实验室 LFP/石墨研究数据，不宣称为济南或企业现场电芯数据。

## 下一步

为本地 `DischargeCapacity.xlsx` 建立人工复核的 17 条条件列布局配置，并实现 `NAUMANN_CYCLE` MATLAB 条件矩阵适配器。后者应只读取明确声明的变量和单位，输出同一条件级退化契约，再由上层将经过核验的数值交给 GP 与主动试验工具。
