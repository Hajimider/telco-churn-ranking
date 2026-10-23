# 学习版调查记录

- 现有 `churn_prediction/` 是已经完成的求职交付版，包含 SQL、训练内五折 CV、概率校准、预算评估、manifest 冻结保护和复杂看板。
- 用户希望保留项目主题，但降低学习门槛，目标是“比入门高、比高级低”的可读版本。
- 现有数据协议需要 `customerID`、电信服务类别、`tenure`、`MonthlyCharges`、`TotalCharges` 和 `Churn` 等字段。
- 学习版默认从自己的 `data/` 目录读取数据，也支持 `--data` 指向已有数据文件；页面应中文显示，机器可读输出保留原始字段名。
- 用户指定验证解释器为 `E:\miniconda\envs\machine-learning\python.exe`。
