# 运行、报告与部署说明

## 环境

Python 3.10；`requirements.txt` 固定了在指定 machine-learning 环境中核验的直接依赖版本，而非整个 Conda 环境的无关包。传递依赖未完全锁定；`metrics.json` 记录运行时主要版本。

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

用户本机解释器为 `E:\miniconda\envs\machine-learning\python.exe`。PyCharm 选择该解释器，工作目录设为仓库根目录，而不是 `telco_churn/`。

## 训练

把 Kaggle 原始 CSV 放入 `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`。默认不需要 MySQL、不调用远程 AI。

完整配置：

```bash
python run_project.py --output outputs/run_validated
```

默认外层 5 折 × 3 重复、内层 3 折、LightGBM 30 trials/外层训练折，LR 的 C 和 RF 的叶节点最小样本数也在内层搜索。所有模型在外层训练数据内部拟合 sigmoid 校准；全训练集再按同一流程拟合最终模型。校准内层使用当前训练子集已选择的超参数，因此校准分数对拟合折外，但没有对超参数选择完全隔离；外层评估仍隔离，应以外层/测试Brier验证校准，不能将校准拟合分数当独立评估。消融使用 LR 家族，四组特征共享外层划分，但不会据消融结果再挑选特征后声称独立验证。

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
| calibration_cv.csv / calibration_comparison.csv | 训练外层/固定测试的校准前后对照 |
| calibration_pr.png / calibration_bins.csv / pr_curve.csv | 校准曲线、PR 图及底层数据 |
| numeric_correlations.csv | 训练集数值相关矩阵，不据此宣称特征独立 |
| segment_summary.csv / segment_inference.csv | 训练分群 Wilson 区间、卡方 Holm、Cramér's V |
| policy_metrics.csv / policy_assignments.csv | 预设10%同容量的历史标签排序对照及名单 |
| budget_training_oof.csv | 只用训练OOF选择名额，包含零联系及全联系 |
| business_scenarios.csv | 3成本 × 3成功率 × 3月数的27个训练OOF情景 |
| budget_sensitivity.csv / budget_curve.png | 测试描述曲线及训练选定K的冻结评估，不从测试集选K |
| logistic_coefficients.csv | 固定预处理坐标下惩罚LR的描述区间，非因果OR |
| shap_summary.png / shap_examples/ | 元数据标注的未校准树基模型，不一定是主模型 |
| ai_customer_context.csv | 仅优先客户的允许字段，不含真实标签 |

月费是收入代理而不是利润或 CLV；收益用历史标签和假设成功率算情景，不是预测的真实挽留收入。常数月值时盈亏平衡精确率 = 单次成本 / (成功率 × 月值 × 剩余月数)。这不构成因果收益模型。

## 看板与可选 AI

```bash
python -m streamlit run dashboard.py
```

看板按最新完整报告自动选取；可手动切换。旧报告继续兼容，但有明确警示；旧截图不代表新版实测结果。

AI 属于可选展示层，不参与训练或排序。未配置时用本地模板；配置 `CHURN_AI_BASE_URL/MODEL/API_KEY/TIMEOUT_SECONDS` 后，点击按钮才发送允许字段。不要在演示真实客户时向第三方发送个人资料，真实标签不发送。AI 输出需要人工核对，不能用于证明模型效果。

Streamlit Community Cloud 的主文件为 `dashboard.py`。上传模块、依赖、字体及同一次完整运行报告；凭据使用平台 Secrets，勿上传 `.env`。云端读取报告，不运行完整训练。

## 验证

```bash
python -m pytest tests telco_churn/test_revision.py -q
```

测试默认不需要 MySQL或远程AI；低预算 fixture 不证明完整配置的实际指标。GitHub Actions 在 Linux/Python 3.10 安装固定直接依赖并运行测试；本地通过不等于远程 CI 已通过。
