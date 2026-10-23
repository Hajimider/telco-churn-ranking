# 电信客户流失学习版设计

日期：2026-10-21

## 目标

在现有求职交付版之外，新增一个适合已经学过 Python、Pandas、基础统计和分类模型，但还没有独立完成完整机器学习项目的读者的学习版。学习版需要让读者理解完整流程：数据读取、数据清洗、特征准备、模型训练、指标评估、结果解释和简单展示。

学习版不追求复刻交付版的所有工程能力，也不使用交付版的冻结报告作为输入。它的结论只用于学习和演示，不代表生产效果、未来时间外泛化或真实挽留收益。

## 方案选择

采用独立项目方案，而不是复制交付版后删除复杂代码。独立目录可以保持清晰的文件边界，避免把 manifest、旧报告、SQL 对账和高级实验的隐含依赖带入学习路径。

学习版保留轻量 Streamlit 看板，因为它能把训练结果、指标和客户风险名单连接起来；看板只读取本项目的 `outputs/`，不在页面交互中重新训练。Notebook 不作为必需入口，避免把运行方式绑定到 IDE。

## 目录和职责

```text
churn_prediction_learning/
├─ README.md
├─ requirements.txt
├─ run_learning.py
├─ dashboard.py
├─ data/README.md
├─ outputs/                 # 运行后生成，默认不提交
├─ tests/test_learning.py
├─ task_plan.md
├─ findings.md
├─ progress.md
└─ docs/superpowers/specs/2026-10-21-learning-version-design.md
```

`run_learning.py` 按教学顺序组织为几个短函数：读取并检查数据、清理 `TotalCharges`、构造少量特征、分层拆分、训练逻辑回归和随机森林、计算指标、保存预测和图片。函数名和注释直接对应 README 的章节。

`dashboard.py` 读取 `outputs/metrics.json`、`outputs/predictions.csv` 和图表数据，展示样本概览、模型指标、混淆矩阵/ROC 图、特征重要性和按风险排序的客户名单。它不会导入交付版模块，也不会读取 `reports/career_final/`。

## 数据和建模范围

输入使用公开 Telco Customer Churn CSV。训练入口支持默认数据路径和 `--data` 参数；缺少文件、缺少关键列、目标列只有一个类别或客户编号重复时给出中文错误。

清洗只做学习所需的最小处理：将空白 `TotalCharges` 转换为缺失并用训练集的中位数填补，类别字段用最常见类别填补并独热编码，数值字段标准化。所有填补、编码和标准化都通过 scikit-learn Pipeline 在训练数据上拟合，避免测试集信息泄漏。

模型固定为：

1. 逻辑回归，作为可解释基线。
2. 随机森林，作为非线性模型对照。

使用一次分层训练/测试拆分，默认 `test_size=0.2`、固定随机种子。报告 Accuracy、Precision、Recall、F1 和 ROC-AUC，并显示混淆矩阵。由于流失标签存在不平衡，README 会提醒不能只看 Accuracy；不会在学习版中引入复杂的重采样、校准或成本敏感决策。

## 输出和看板

每次运行输出到用户指定的 `--output` 目录，默认 `outputs/`。如果目录已有结果，程序提示换目录或清理学习版自己的输出，不允许覆盖现有求职交付版目录。

输出最少包含：`metrics.json`、`predictions.csv`、`confusion_matrix.png`、`roc_curve.png`、`feature_importance.csv`。预测 CSV 保留 `customerID`、真实标签、两个模型的概率和预测标签，便于读者用 Pandas 自己检查。

README 提供两条路径：先运行 CLI 生成结果，再启动看板查看结果。每一步说明输入、输出、常见错误和指标含义。页面表头和图表标题默认中文，标准模型指标保留常用缩写并配中文解释。

## 测试和验收

测试使用项目内合成小数据，不依赖公开 CSV 下载，覆盖：

- 清洗后可训练并生成两个模型结果；
- 训练/测试拆分保持类别存在；
- 输出文件存在且指标字段完整；
- 缺少关键列、重复客户编号和单一目标类别给出明确错误；
- 看板 AppTest 无异常并显示中文表头。

真实数据验收使用用户指定解释器执行一次 CLI，确认结果目录生成；再启动看板或用 AppTest 读取该目录。验收不要求与交付版的 AP、预算命中或模型选择结果相同。

## 明确不做的内容

学习版不实现 SQL 分群对账、九组候选模型、五折交叉验证、训练内概率校准、冻结 manifest、模型哈希、SHAP、bootstrap 区间、真实收益敏感性分析、线上实验设计和生产部署。需要这些能力时，回到现有 `churn_prediction/` 求职交付版阅读。
