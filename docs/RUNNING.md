# 运行、报告与部署说明

## 环境

Python 3.10；`requirements.txt` 只包含运行时依赖，测试工具在 `requirements-dev.txt`。传递依赖未完全锁定；`metrics.json` 记录运行时主要版本。

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
# 开发和测试环境额外安装：
python -m pip install -r requirements-dev.txt
```

使用已安装项目依赖的 Python 3.10 环境；PyCharm 工作目录设为仓库根目录，而不是 `telco_churn/`。

## 训练

把 Kaggle 原始 CSV 放入 `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`。默认不需要 MySQL、不调用远程 AI。

完整配置：

```bash
python run_project.py --output outputs/run_new
```

默认外层 5 折 × 3 重复，校准交叉拟合 3 折，LR/RF 内层 3 折搜索，LightGBM 每次搜索 30 trials。每个外层训练折先搜索一次超参数，再在固定超参下交叉拟合 sigmoid；最终估计器在当前完整训练子集内重新搜索并拟合。外层验证样本不参与调参、校准或预处理。代码根据每个模型的外层 Brier 决定最终推理使用原始模型或校准模型，并让阈值、训练 OOF 与预算分析使用同一尺度的有效概率。该完整嵌套会增加训练时间；校准效果仍须以未参与拟合的外层预测评估。

历史留出集 Bootstrap 区间覆盖 AP、ROC-AUC、Brier、固定 0.5 阈值指标及固定 10% 名额 Precision@K/Recall@K。区间条件于已拟合模型，只描述当前历史留出样本的抽样波动，不包括训练和选模不确定性。固定留出集已用于开发复核，不是独立盲测；嵌套交叉验证结果作为主要泛化证据。消融使用 LR 家族，四组特征共享外层划分，但不会据消融结果再挑选特征后声称独立验证。

各模型的外层分数用于模型比较，按同一批外层结果选出的最高分模型仍会有模型选择效应；该均值不应解释为独立嵌套模型选择校正后的最终性能估计。

低预算检查（不能替代完整实验结果）：

```bash
python run_project.py --output outputs/smoke_new --outer-splits 3 --repeats 1 --inner-splits 2 --trials 1 --bootstrap 20 --no-explanations
```

输出目录必须位于项目内且为空。已有目录不覆盖；中断后另选目录。`metrics.json` 最后写入，看板不读取半成品。

## MySQL（可选）

默认 SQLite 内存数据库直接执行 `sql/segment_analysis.sql`，无需另装服务器。显式 MySQL 不静默回退，便于发现配置错误。

将 `.env.example` 复制为 `.env`，填写 `CHURN_MYSQL_HOST/PORT/DATABASE/USER/PASSWORD`。凭据只留本机；不要提交 `.env`。进程环境变量优先。

先在自己的 MySQL 8.0+ 中创建目标库并授权会话临时表，再检查连接：

```bash
python -m telco_churn.mysql_segments
```

使用 MySQL 重跑：

```bash
python run_project.py --backend mysql --output outputs/run_mysql_new
```

不会替你创建持久化业务表或清空已有表；总体、三个分群及交叉分群一次会话计算。快照没有行为时间，不能伪造留存 cohort、LAG 用户日志或滑窗特征。

## 报告文件

| 文件 | 口径 |
|---|---|
| metrics.json | 配置、主模型、测试指标、模型比较与 Git dirty 状态 |
| nested_cv_folds.csv | 每次外层评估的 AP/AUC、校准前后 Brier |
| training_oof.csv | 重复外层预测按客户平均；用于选阈值和情景名额 |
| ablation_folds.csv / ablation_summary.csv | LR 四组特征的逐折 AP 与折间标准差 |
| model_comparison_test.csv | Nadeau–Bengio 近似修正 + Holm p 值，区间为逐比较区间 |
| calibration_cv.csv / calibration_comparison.csv | 训练外层/历史留出集校准对照 |
| calibration_pr.png / calibration_bins.csv / pr_curve.csv | 校准曲线、PR 图及底层数据 |
| numeric_correlations.csv | 训练集数值相关矩阵，不据此宣称特征独立 |
| segment_summary.csv / segment_inference.csv | 训练分群 Wilson 区间、卡方 Holm、Cramér's V |
| policy_metrics.csv / policy_assignments.csv | 预设10%同容量的历史标签排序对照及名单 |
| budget_training_oof.csv | 只用训练OOF、按风险分数选择名额，包含零联系及全联系 |
| budget_training_oof_value.csv | 只用训练OOF、按风险分数×月费价值代理排序的对照 |
| business_scenarios.csv | 3成本 × 3成功率 × 3月数的27个训练OOF情景，含风险排序与价值排序 |
| budget_sensitivity.csv / budget_curve.png | 风险排序的留出集描述曲线及训练选定K的冻结评估，不从留出集选K |
| budget_sensitivity_value.csv / budget_curve_value.png | 价值排序的对应曲线与冻结评估 |
| segment_pareto.csv | 按流失月费敞口排序的累计分群结果 |
| logistic_coefficients.csv | 固定预处理坐标下惩罚LR的描述区间、VIF 与共线性提示，非因果OR |
| ablation_summary.csv | 四组特征消融的均值、配对差值区间与 Holm 校正 p 值 |
| shap_summary.png / shap_examples/ | 元数据标注的未校准树基模型，不一定是主模型 |
| ai_customer_context.csv | 仅优先客户的允许字段，不含真实标签 |

月费是收入代理而不是利润或 CLV；收益用历史标签和假设成功率算情景，不是预测的真实挽留收入。常数月值时盈亏平衡精确率 = 单次成本 / (成功率 × 月值 × 剩余月数)。这不构成因果收益模型。

## 看板与可选 AI

```bash
python -m streamlit run dashboard.py
```

看板按最新完整报告自动选取；可手动切换。当前权威报告为 `outputs/run_final`。

AI 属于可选展示层，不参与训练或排序。未配置时用本地模板；配置 `CHURN_AI_BASE_URL/MODEL/API_KEY/TIMEOUT_SECONDS` 后，点击按钮才发送允许字段。不要在演示真实客户时向第三方发送个人资料，真实标签不发送。AI 输出需要人工核对，不能用于证明模型效果。

Streamlit Community Cloud 的主文件为 `dashboard.py`。上传模块、依赖、字体及同一次完整运行报告；凭据使用平台 Secrets，勿上传 `.env`。云端读取报告，不运行完整训练。

## 验证

```bash
python -m pytest tests telco_churn/test_revision.py -W error -q --tb=short
```

测试默认不需要 MySQL或远程AI；低预算 fixture 不证明完整配置的实际指标。当前代码回归为 53 项测试且要求零警告；本地通过不等于远程 CI 已通过。
