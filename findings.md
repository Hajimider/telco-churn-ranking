# 学习版调查记录

- 现有 `churn_prediction/` 是已经完成的求职交付版，包含 SQL、训练内五折 CV、概率校准、预算评估、manifest 冻结保护和复杂看板。
- 用户希望保留项目主题，但降低学习门槛，目标是“比入门高、比高级低”的可读版本。
- 现有数据协议需要 `customerID`、电信服务类别、`tenure`、`MonthlyCharges`、`TotalCharges` 和 `Churn` 等字段。
- 学习版默认从自己的 `data/` 目录读取数据，也支持 `--data` 指向已有数据文件；页面应中文显示，机器可读输出保留原始字段名。
- 用户指定验证解释器为 `E:\miniconda\envs\machine-learning\python.exe`。

## 2026-10-25 AI 增强规划发现

- 现有看板只读取报告目录，不读取原始 CSV；AI 页面必须由训练流程额外导出白名单客户上下文，不能在页面临时拼接测试标签。
- 现有 policy_assignments.csv 含有 actual_churn，该字段只能用于离线策略评估，必须从 AI 提示词和 AI 页面上下文中排除。
- 项目已有 pandas、scikit-learn、streamlit 和标准库，OpenAI 兼容 HTTP 调用可用 urllib.request 实现，不需要新增必需 SDK。
- 推荐采用“模板优先、接口可选”的 AI 方案，以保证无密钥、无网络时仍可演示；模型生成只负责解释和建议，不修改风险分数、阈值或名单。
- 本轮规划未执行代码、未新增依赖、未调用外部模型接口，也未修改高级项目。
