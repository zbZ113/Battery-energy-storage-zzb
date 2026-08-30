# ADR-0002：数据集必须按 cell_id 隔离

- 状态：Accepted
- 日期：2026-07-28

## 背景

一个电芯包含多行、多个循环和大量采样点。若按行、循环或窗口随机划分，同一电芯会同时
进入训练和测试，模型可以识别个体而非学习可泛化的退化规律，指标因此失真。

## 决策

1. train、validation、calibration 和 test 必须以 `cell_id` 为最小隔离单位；
2. 划分清单必须版本化并记录 SHA-256；
3. 训练、早停、超参数选择、Conformal 校准和最终测试分别使用冻结集合；
4. cutoff 或特征窗口变化不能改变电芯归属；
5. 校准目标电芯不能同时出现在 calibration cohort；
6. 新数据集必须重新冻结并审计划分，不继承 MATR 身份。

## 备选方案

- 按采样点或循环随机拆分：严重泄漏，拒绝；
- 每次训练临时随机拆分：难以对账和复现，拒绝；
- 按批次整体拆分：可用于专门的跨批次实验，但不能替代当前固定主评估协议；
- 时间序列前后切分同一电芯：可用于个体在线更新研究，但不能作为独立电芯泛化测试。

## 后果

正面：

- 测试指标反映未见电芯泛化；
- calibration 与 test 相互独立；
- 80 次正式运行可以逐坐标对账。

代价：

- 样本量按电芯数计算，统计不确定性更明显；
- 新增或删除电芯会改变数据版本；
- 不能把单个电芯的数千行误称为数千个独立样本。

## 实现证据

- `DATA_CONTRACT.md`；
- `EXPERIMENT_PROTOCOL.md`；
- `src/quanxin_life/data/` 与 `src/quanxin_life/training/`；
- `src/quanxin_life/evaluation/`；
- 团队本地维护的 split、training 和 calibration 契约回归记录（不随 Git 分发）。

正式 MATR 固定划分及样本数见 [项目状态](../status.md)。

## 成熟度

- MATR 三批固定划分：Validated；
- 未见外部或企业数据划分：Planned，需独立数据许可与新清单；
- 删失感知评估：Planned。
