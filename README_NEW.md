# 电信客户流失预测与排序

使用 7,043 条电信客户数据，构建流失预测模型并在固定联系名额下排序客户。**核心方法：** LightGBM + Optuna 调参 + SHAP 解释 + 生存分析 + 名额敏感性曲线

## 核心结果

- **模型性能：** LightGBM (AP=0.71, 95% CI: 0.68-0.74) 优于逻辑回归 (AP=0.66) 和随机森林 (AP=0.65)，配对 t 检验 p<0.05
- **排序效果：** 10% 名额下命中率 78%，覆盖 28.9% 的流失客户，是随机基线的 2.9 倍
- **生存分析：** Cox 模型显示月付合同的流失风险是两年合同的 14.2 倍 (HR=14.2, 95% CI: 10.1-20.0, p<0.001)
- **业务优化：** 最优联系名额为 12%，期望净收益达 4.8 万元（假设挽留成本 50 元/人，挽留成功率 30%）

## 方法与数据

### 数据集
- 来源：[Kaggle Telco Customer Churn](https://kaggle.com/datasets/blastchar/telco-customer-churn)
- 规模：7,043 条客户记录，流失率 26.5%
- 划分：训练集 5,634 条，测试集 1,409 条（分层抽样，随机种子 42）

### 特征工程（30+ 特征）
原始 19 个字段 → 衍生特征：
- **费用相关：** 实际月均与当前月费偏离、高费用标记
- **服务组合：** 增值服务开通数、是否全套保护服务
- **客户生命周期：** 新客户/忠诚客户标记、合同类型 × 在网时长交叉
- **支付行为：** 自动支付标记、无纸化账单 × 电子支票风险组合

### 模型训练
1. **候选模型：** 逻辑回归、随机森林、LightGBM
2. **超参数优化：** Optuna 对 LightGBM 进行 100 轮 Bayesian 优化，最大化 5 折 CV 的 AP
3. **类别不平衡处理：** `scale_pos_weight` 根据训练集流失率自动调整
4. **模型选择：** 基于训练集 5 折交叉验证 AP 均值，LightGBM 胜出

### 统计分析
- **客户分群：** SQL 分群统计 + 卡方独立性检验 + Holm 多重比较校正 + Cramér's V 关联强度
- **生存分析：** Kaplan-Meier 曲线（按合同类型分层）+ Log-rank 检验 + Cox 比例风险模型
- **置信区间：** Bootstrap (1000 次) 估计测试集指标的 95% 置信区间
- **显著性检验：** 配对 t 检验比较模型的交叉验证 AP 差异

### 模型解释
- **逻辑回归：** 输出系数、OR 值及 Bootstrap 95% 置信区间
- **LightGBM：** SHAP 值分析，包括 summary plot 和 top 客户 waterfall plot

### 业务分析
- **固定名额排序：** 10% 名额下的 Precision@K、Recall@K、Lift@K
- **名额敏感性：** 评估 5%-50% 名额下的指标变化
- **收益模型：** 期望净收益 = 挽留成功价值 - 联系成本

## 主要结果

### 模型比较

| 模型         | 五折 AP 均值 | AP 95% CI    | 测试集 AP | 测试集 F1 | ROC-AUC |
|:------------ | ------------:| ------------:| ---------:| ---------:| -------:|
| **LightGBM** | **0.708**    | 0.68-0.74    | 0.713     | 0.627     | 0.851   |
| 逻辑回归     | 0.661        | 0.64-0.69    | 0.633     | 0.604     | 0.842   |
| 随机森林     | 0.650        | 0.63-0.68    | 0.640     | 0.571     | 0.837   |

**模型比较检验：** LightGBM vs 逻辑回归，配对 t 检验 t=3.42, p=0.027，差异显著。

### 排序表现（10% 名额）

| 排序方式       | 命中流失人数 | Precision@K | Recall@K | Lift@K |
|:-------------- | ------------:| -----------:| --------:| ------:|
| **LightGBM**   | **108**      | **77.1%**   | **28.9%**| **2.91**|
| 逻辑回归       | 104          | 74.3%       | 27.8%    | 2.80   |
| 随机森林       | 108          | 77.1%       | 28.9%    | 2.91   |
| 短在网时长规则 | 87           | 62.1%       | 23.3%    | 2.34   |
| 随机期望       | 37.2         | 26.5%       | 9.9%     | 1.00   |

### 生存分析

**Kaplan-Meier 中位生存时间：**
- 月付合同：29 个月
- 一年合同：72 个月（未达到）
- 两年合同：72 个月（未达到）

**Log-rank 检验：** χ²=523.4, df=2, p<0.001，不同合同类型的生存曲线差异显著。

**Cox 模型 Top 5 风险因素：**

| 因素                    | HR    | 95% CI        | p值    |
|:----------------------- | -----:| -------------:| ------:|
| 月付合同                | 14.2  | 10.1 - 20.0   | <0.001 |
| 光纤互联网              | 2.8   | 2.1 - 3.7     | <0.001 |
| 无技术支持              | 2.3   | 1.8 - 2.9     | <0.001 |
| 电子支票支付            | 1.9   | 1.5 - 2.4     | <0.001 |
| 月费（每增加 10 元）    | 1.3   | 1.2 - 1.4     | <0.001 |

### 名额敏感性与收益

最优名额比例为 **12%**，此时：
- Precision@K = 74.5%
- Recall@K = 31.2%
- **期望净收益 = 4.8 万元**

假设：挽留成本 50 元/人，客户月均价值 60 元，平均剩余生命周期 12 个月，挽留成功率 30%。

## 运行项目

### 环境要求
Python 3.10+，依赖安装：

```bash
pip install -r requirements.txt
```

### 数据准备
从 [Kaggle](https://kaggle.com/datasets/blastchar/telco-customer-churn) 下载 `WA_Fn-UseC_-Telco-Customer-Churn.csv`，放入 `data/` 目录。

### 训练模型

```bash
python run_project.py --output outputs/run_optimized
```

可选参数：
- `--test-size`：测试集占比（默认 0.2）
- `--random-state`：随机种子（默认 42）

训练包含：特征工程、LightGBM 调参（100 轮）、模型评估、SHAP 解释、生存分析、名额敏感性分析，预计耗时 10-15 分钟。

### 查看看板

```bash
streamlit run dashboard.py
```

仓库自带演示报告（`outputs/project_v2/`），无需训练即可查看。

## 项目结构

```
├── churn_pipeline.py           # 主流程：数据处理、模型训练、评估
├── lightgbm_tuning.py          # Optuna 超参数优化
├── model_explanation.py        # SHAP 和逻辑回归系数解释
├── survival_analysis.py        # Kaplan-Meier、Cox 模型
├── statistical_inference.py    # Bootstrap CI 和显著性检验
├── budget_analysis.py          # 名额敏感性与收益分析
├── dashboard.py                # Streamlit 看板
├── sql/segment_analysis.sql    # 客户分群 SQL
├── tests/test_project.py       # 单元测试
└── outputs/                    # 训练输出
    ├── metrics.json            # 模型指标、超参数、统计检验
    ├── logistic_coefficients.csv
    ├── shap_summary.png
    ├── survival_curves.png
    ├── cox_results.csv
    ├── budget_sensitivity.csv
    └── budget_curve.png
```

## 关键技术点

1. **特征工程：** 从 19 个原始特征扩展到 30+ 特征，提升模型表现
2. **调参优化：** Optuna Bayesian 优化，AP 从 0.66 提升到 0.71
3. **统计严谨性：** Bootstrap 置信区间、配对 t 检验、多重比较校正
4. **生存分析：** 将流失预测与生存时间结合，识别风险因素
5. **业务导向：** 名额敏感性曲线和收益模型，支持决策优化
6. **可解释性：** SHAP 值 + 逻辑回归 OR 值，模型透明可信

## 数据限制

本项目基于历史快照数据，缺少事件时间、未来观察窗口和实际干预记录。结果仅说明模型对历史流失标签的区分和排序能力，不代表未来流失预测或挽留效果。模型分数未经概率校准，不应直接解释为真实流失概率。分群统计反映关联，不代表因果关系。

## 扩展方向

- **Uplift 建模：** 在有随机对照试验数据时，建模干预的因果效应（CATE）
- **时间维度：** 使用带时间窗口的数据，评估模型的时间稳定性
- **特征消融：** 系统评估每类特征的边际贡献
- **深度学习：** 尝试 TabNet、DeepFM 等深度模型

## 参考资料

- [Kaggle Telco Customer Churn](https://kaggle.com/datasets/blastchar/telco-customer-churn)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Lifelines: Survival Analysis](https://lifelines.readthedocs.io/)
- [Optuna: Hyperparameter Optimization](https://optuna.org/)
