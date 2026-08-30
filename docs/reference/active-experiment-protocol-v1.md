# 主动试验推荐协议 v1

## 1. 目标与边界

`quanxin_life.experiments.gp_active` 是面向低维储能电芯工况的确定性数值服务。它只对调用方显式提供、已经完成来源审查的观测数据拟合高斯过程（GP），并在已审核运行边界内排序候选试验。

模块不读取 Naumann 原始文件，不制造 Naumann 或企业数据，不加载 pickle、joblib、`.pt`、`.pth` 等序列化制品，也不通过 LLM 生成退化率、方差、误差或推荐分数。数据接入、来源验证和正式 `ToolResult` 封装由上层受治理工具完成；Agent 只能调用该工具，不能自行重算或改写数值。

该模块的输出是**试验优先级启发式结果**，不是工业安全批准、生产部署结论或任何策略“必然优于”其他策略的声明。

## 2. 输入契约

每条已完成观测必须显式包含：

- `temperature_c`；
- `mean_soc`；
- `dod`；
- `charge_c_rate`；
- `discharge_c_rate`；
- 单一、真实观测到的目标名称与数值，例如 `capacity_loss_rate`、`equivalent_cycle_degradation_rate` 或 `resistance_growth_rate`；
- 试验时长和设备成本。

调用方还必须给出人工审核过的五维工况上下界，以及时长、设备成本的归一化基准。边界代表当前设备和安全流程已允许的候选空间，不是电化学安全结论。缺失、非有限、混合目标名、重复观测标识或越界观测都会被拒绝；模块不插补、不裁剪候选条件到边界。

每个**待推荐候选**还必须显式给出 `safety_approved=True` 与 `equipment_available=True`。这两个字段默认是 `None`，不会默认视作已审批或设备可用：缺失时分别返回 `SAFETY_APPROVAL_MISSING`、`EQUIPMENT_AVAILABILITY_MISSING`；显式为 `False` 时分别返回 `SAFETY_NOT_APPROVED`、`EQUIPMENT_UNAVAILABLE`。

特征顺序固定为：

```text
temperature_c, mean_soc, dod, charge_c_rate, discharge_c_rate
```

在该定义下，`mean_soc` 与 `dod` 还必须共同形成有效 SOC 窗口：

\[
\mathrm{soc\_lower}=\mathrm{mean\_soc}-\frac{\mathrm{dod}}{2}\geq 0,
\qquad
\mathrm{soc\_upper}=\mathrm{mean\_soc}+\frac{\mathrm{dod}}{2}\leq 1
\]

任何违反该联合约束的候选都会以 `SOC_WINDOW_OUT_OF_RANGE` 拒绝，并给出 `SOC_LOWER_BELOW_ZERO` 或 `SOC_UPPER_ABOVE_ONE` 细节；模块不会通过裁剪 SOC 或 DOD 来让其通过。

## 3. GP 模型

模型使用 `StandardScaler` 后的 scikit-learn `GaussianProcessRegressor`，内核固定为：

```text
Matérn(ν=2.5) + WhiteKernel
```

随机种子固定为 `20260712`。首版关闭隐式超参数优化：小样本工况数据中的边界最优核参数不能直接视为可靠工程结论。模型只驻留内存，不经 pickle、joblib、`.pt`、`.pth` 读写。

## 4. 主策略：成本约束的 EIVR

默认主策略是**期望积分后验方差下降**（Expected Integrated Variance Reduction，EIVR），而不是单点预测标准差。

调用 `rank_candidates` 或 `recommend_batch` 时，调用方必须明确传入非空、去重、边界内的固定 `reference_conditions`。该参考空间在一次排序或有限池回放期间保持不变。对每个安全且设备可用的候选条件 \(x\)，模块只利用 GP 协方差解析更新：

\[
\Delta V(x)=\sum_{r\in R}\frac{\operatorname{Cov}(r,x\mid D)^2}
{\operatorname{Var}(x\mid D)}
\]

其中 \(R\) 是固定参考空间、\(D\) 是已观测实验。分子使用目标函数的后验协方差；分母加入 `WhiteKernel` 的观测噪声，因此不会把一次带噪试验伪装成能够消除全部模型不确定性。协方差和方差下降会还原到目标变量量纲后再写入审计元数据。该计算不需要也不允许伪造“若执行该候选试验后”的观测目标。候选分数为：

\[
\frac{\Delta V(x)}
{\mathrm{duration}(x)/T_0+\mathrm{equipment\_cost}(x)/C_0}
-\mathrm{duplicate\_penalty}(x)
\]

每个被接收候选都会返回以下可审计数值元数据：

- 策略名称；
- 参考条件数量；
- 参考空间的更新前总后验方差；
- 假设观测该候选后的总后验方差；
- 两者的方差下降量；
- 归一化成本与重复惩罚；
- 最终采集分数。

`MAX_VARIANCE` 仅保留为显式基线：其分数就是单点预测标准差（等价地可使用单点预测方差），**不**除以成本，也**不**扣相似性惩罚。成本、EIVR 与相似重复惩罚只属于 `COST_AWARE_EIVR`。

成本约束 EIVR 的相似重复惩罚根据候选与已观测、已选条件在审核边界归一化空间的最近相似度计算。若候选与任一已观测条件五维向量完全相同，则无论 `duplicate_penalty_weight` 是否为零，都会拒绝为 `DUPLICATE_OBSERVED_CONDITION`，而不是静默推荐；相近但不等价的候选才使用配置化相似性惩罚。

## 5. 批量推荐与去重

`recommend_batch` 逐个选择候选。每选中一个条件，后续与其五维条件向量完全相同的候选会被明确拒绝，拒绝码为：

```text
DUPLICATE_SELECTED_CONDITION
```

相近但不等价的候选仍可以参与成本约束 EIVR 排序，不过会受到已选条件带来的重复惩罚；`MAX_VARIANCE` 不使用该惩罚。这样既避免同一工况在一个批次中被重复接受，也不会把近邻工况误判为同一个条件。

对于批量 `COST_AWARE_EIVR`，每选中一个尚未执行、尚未揭示标签的条件 \(s\)，系统使用结果无关的协方差条件更新后再计算下一轮：

\[
C_{\mathrm{next}}(a,b)=C(a,b)-C(a,s)
\left[C(s,s)+\sigma^2_{\mathrm{noise}}\right]^{-1}C(s,b)
\]

这里仅使用已选的条件向量与 GP `WhiteKernel` 噪声；不会读取、推断或伪造该候选的退化标签。候选越界、未获安全批准或设备不可用同样以明确拒绝码返回，不会被预测、评分或裁剪。

## 6. 有限候选池回放

`evaluate_finite_pool_replay` 只接受调用方提供的、带真实或明确标记为合成标签的有限观测池。它固定初始已揭示观测、固定查询预算与参考空间，并在同一预算下依次比较四种策略：

1. `RANDOM`：基于固定种子的有限池随机选择；
2. `UNIFORM_GRID`：在已有揭示条件的归一化空间中选择最远候选，作为有限池空间填充/网格基线；
3. `MAX_VARIANCE`：单点预测标准差基线；
4. `COST_AWARE_EIVR`：成本约束 EIVR 主策略。

每轮先用已揭示标签拟合 GP，选择一个未揭示条件，记录该条件的预测均值、预测标准差和绝对误差，随后才揭示其调用方提供的实际标签并进入下一轮。每条轨迹报告累计 MAE、累计 RMSE、累计预测标准差均值和训练样本数。

回放池必须拒绝重复 `observation_id`，也必须拒绝任意两条记录具有完全相同的五维工况向量；错误信息会同时列出重复条件和对应的观测 ID。这样不会把同一工况的重复标签带入“未揭示候选”池，并在后续误报为笼统的无候选运行时错误。

四种回放策略的**选样 ID 轨迹**只依赖固定核的协方差、候选工况、成本、审核边界和固定随机种子；未揭示候选的目标标签不参与选样。标签仅在其条件已经选中后用于记录预测误差。因此，在固定初始观测和预算下，仅修改未揭示候选标签不得改变任一步的选样 ID 轨迹；这是一项防泄漏回归约束，并不代表预测误差也不变。

回放结果不产生胜者字段，也不宣称某策略优于随机、网格或任何其他方法。是否存在优势只能由固定数据、固定成本模型、固定预算和独立实验报告共同决定。

## 7. 后验协方差数值稳定性

虚拟条件更新和同一条件空间的后验协方差会显式对称化。仅当对角线落在 `[-1e-10, 0)` 的明确浮点舍入容差内时，才将该极小负方差设为零；任何更明显的负方差、非有限协方差或不可解的虚拟观测协方差都会以错误停止，不会静默裁剪或伪造不确定性。

## 8. 集成边界

后续 `recommend_next_experiment` 工具必须把数据来源、模型版本、边界版本、成本配置版本、输入哈希、数值输出和拒绝原因封装为 `ToolResult`。正式报告只可引用该受审计工具的 `result_id`；不得直接将本模块的内存对象或 Agent 文本写入正式数值结论。
