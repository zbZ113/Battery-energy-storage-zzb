# 短期物理核验工具协议 v1

`check_operating_condition` 是 `PhysicsValidationRequest` 到共享 `ToolResult` 的唯一工具绑定。
它调用短时 SPMe + Prada2013 工况核验，不生成长期退化轨迹、EOL、RUL、寿命标签或质保结论。

## 运行边界

- PyBaMM 是可选依赖；未安装、加载失败或求解失败时，工具返回原始的 `UNAVAILABLE` 或 `FAILED`
  状态，样本数组为空，不用规则或模型伪造替代数值。
- 完成态的电压、电流、放电容量、SOC 与终止原因均直接来自物理求解器。
- 每次结果都保留 SPMe、Prada2013、PyBaMM 版本、实验程序、边界风险和强制的“参数未针对目标
  电芯标定，仅供短期趋势与边界核验”警告。
- `uncertainty.short_horizon_only=true` 是边界声明，不是统计置信区间。

所有调用方必须通过共享注册表使用该工具，并将返回的 `result_id` 作为报告中的物理参考来源。
物理核验状态不是寿命模型的校准证据；任何长期寿命结论必须另有电芯级模型和不确定性来源。
