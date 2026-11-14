# 电信客户流失项目优化计划

> 本文保留原优化需求与验收设想，不作为当前完成状态或实测指标的来源。项目说明与运行方式见 [README](../README.md)；训练主流程现位于 `telco_churn/pipeline.py`，各分析模块也已归入 `telco_churn/`。

**目标：** 将项目从学习项目提升为适合数据挖掘实习岗的有竞争力的简历项目

**时间预估：** 3-5 天（每天 4-6 小时）

---

## 第一阶段：修复致命问题（优先级：紧急，0.5 天）

### 1.1 修正 Git 提交信息不实问题

**问题：** 最新提交称"增加时间稳定性与特征消融评估"，但代码中没有这些内容

**方案：**
- [ ] 删除或修正误导性的提交描述
- [ ] 整理提交历史，移除 `docs/superpowers/` AI 工具痕迹
- [ ] 统一提交信息格式，使用规范的中文描述

**操作：**
```bash
# 方案 A：修正描述（推荐）
git commit --amend -m "添加统计推断：Wilson 区间、卡方检验、Holm 校正与折外阈值选择"

# 方案 B：真正补上时间稳定性和消融评估（见第三阶段）
```

**时间：** 0.5 天

---

## 第二阶段：核心建模能力提升（优先级：最高，2 天）

### 2.1 特征工程（0.5 天）

**当前问题：** 原始字段直接进模型，没有衍生特征

**新增特征：**
- [ ] **费用相关**
  - `charges_deviation`: `TotalCharges / tenure - MonthlyCharges`（实际月均与当前月费的偏离）
  - `charges_per_service`: `MonthlyCharges / service_count`
  - `has_high_charges`: `MonthlyCharges > 70` 的二值标记
  
- [ ] **服务组合**
  - `service_count`: 增值服务开通数（OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies）
  - `has_internet`: 是否有互联网服务
  - `full_protection`: OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport 是否全开
  
- [ ] **客户价值**
  - `tenure_contract_interaction`: tenure × Contract 的交叉特征
  - `is_new_customer`: tenure ≤ 6 个月
  - `is_loyal_customer`: tenure ≥ 36 个月

**实现位置：** `telco_churn/pipeline.py` 在 `prepare_data` 后新增 `engineer_features` 函数

**验收标准：** 特征数从 19 个增加到 30+ 个

---

### 2.2 引入 LightGBM 并调参（1 天）

**当前问题：** 只有 LR 和 RF，缺少 GBDT 系列模型；无调参

**方案：**
- [ ] 添加 LightGBM 到候选模型
- [ ] 使用 Optuna 做超参数优化（50-100 轮）
- [ ] 优化目标：最大化 5 折 CV 的 AP 均值
- [ ] 处理类别不平衡：`scale_pos_weight` 或 `class_weight`

**调参空间：**
```python
{
    'n_estimators': (100, 500),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3, log),
    'num_leaves': (20, 100),
    'min_child_samples': (5, 50),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0.0, 10.0),
    'reg_lambda': (0.0, 10.0),
}
```

**期望结果：** LightGBM 的 AP 提升到 0.68-0.72（比现在的 0.661 提升 2-8 个百分点）

**新增依赖：**
```txt
lightgbm>=4.0
optuna>=3.0
```

**实现位置：** `telco_churn/lightgbm_tuning.py` 提供 `tune_lightgbm`，由 `telco_churn/pipeline.py` 调用并保存最优超参到 `metrics.json`

---

### 2.3 模型可解释性（0.5 天）

**当前问题：** 主模型是 LR，却只输出 RF 的特征重要性

**方案：**
- [ ] **逻辑回归：** 输出系数、OR 值（exp(coef)）及其 95% 置信区间
  - 使用 `model.coef_` 和 bootstrap 估计 CI
  - 保存到 `logistic_coefficients.csv`
  
- [ ] **LightGBM：** SHAP 值分析
  - 使用 `shap.TreeExplainer`
  - 保存 summary plot 到 `shap_summary.png`
  - 保存 top 20 客户的 waterfall plot 到 `shap_examples/`
  
- [ ] **Dashboard 展示：** 新增"模型解释"页，展示系数表和 SHAP 图

**新增依赖：**
```txt
shap>=0.42
```

**实现位置：** `telco_churn/model_explanation.py` 提供模型解释函数，由 `telco_churn/pipeline.py` 调用

---

## 第三阶段：统计专业优势（优先级：高，1 天）

### 3.1 生存分析（0.5 天）

**当前问题：** `tenure` 是天然的生存时间，没有用上生存分析

**方案：**
- [ ] **Kaplan-Meier 曲线：** 按 Contract 类型分层
- [ ] **Log-rank 检验：** 比较不同合同类型的生存曲线差异
- [ ] **Cox 比例风险模型：** 识别显著的风险因素及 HR 值

**产出：**
- `survival_curves.png`：KM 曲线图
- `cox_results.csv`：HR 值及置信区间
- README 中增加"生存分析"章节

**新增依赖：**
```txt
lifelines>=0.28
```

**实现位置：** `telco_churn/survival_analysis.py` 提供 `survival_analysis`，由 `telco_churn/pipeline.py` 调用

---

### 3.2 统计推断增强（0.5 天）

**当前问题：** 测试集指标没有置信区间；卡方检验覆盖维度少

**方案：**
- [ ] **Bootstrap 置信区间：** 对测试集 AP、F1、Precision、Recall 做 1000 次 bootstrap，报告 95% CI
- [ ] **显著性检验：** LightGBM vs LR 的 AP 差异做配对 t 检验（基于 5 折 CV）
- [ ] **卡方扩展：** 增加更多分群维度（PaymentMethod、InternetService、TechSupport 等）

**产出：**
- `metrics.json` 中每个指标增加 `ci_lower` 和 `ci_upper`
- `model_comparison_test.csv`：配对检验结果

**实现位置：** `telco_churn/statistical_inference.py` 提供 `bootstrap_metrics`，由 `telco_churn/pipeline.py` 的 `run_pipeline` 调用

---

## 第四阶段：业务分析深度（优先级：中，1 天）

### 4.1 名额敏感性分析（0.5 天）

**当前问题：** 10% 名额是拍定的，没有做敏感性曲线

**方案：**
- [ ] 计算不同 K 下的 Precision@K、Recall@K、Lift@K（K = 5%, 10%, 15%, ..., 50%）
- [ ] 绘制 Precision-Recall-K 曲线
- [ ] 加入简单的收益模型：
  - 假设挽留成本 50 元/人，客户月均价值 60 元，平均剩余生命周期 12 个月
  - 挽留成功率假设 30%
  - 计算不同 K 下的期望净收益

**产出：**
- `budget_sensitivity.csv`
- `budget_curve.png`
- README 增加"名额与收益分析"章节

**实现位置：** `telco_churn/budget_analysis.py` 提供 `budget_sensitivity_analysis`，由 `telco_churn/pipeline.py` 调用

---

### 4.2 Uplift 建模讨论（0.5 天）

**方案：**
- [ ] 在 README 增加"方法局限与改进方向"章节
- [ ] 讨论当前方法（churn propensity）与 uplift modeling 的区别
- [ ] 说明在有随机对照试验数据时，如何建模 CATE（因果森林、S-learner、T-learner）

**不实现代码，仅文档**

**参考资料：**
- Gutierrez & Gérardy (2017): Causal Inference and Uplift Modelling
- Uber's causal ML library

---

## 第五阶段：工程与展示优化（优先级：中低，0.5-1 天）

### 5.1 简化 AI 模块（0.3 天）

**方案：**
- [ ] 将 AI 解读降级为可选附属功能，README 中移到末尾
- [ ] Dashboard 首页移除 AI 相关内容，聚焦模型结果
- [ ] `ai_assistant.py` 压缩到 100 行以内，或移到独立子目录

---

### 5.2 README 重写（0.2 天）

**当前问题：** 免责声明过多，缺少一句话亮点

**新结构：**
```markdown
# 电信客户流失预测与排序

使用 7,043 条公开电信客户数据，构建流失预测模型并在固定联系名额下排序客户。
**核心方法：** LightGBM + Optuna 调参 + SHAP 解释 + 生存分析 + 名额敏感性曲线

## 核心结果（一句话）

- LightGBM (AP=0.71) > 逻辑回归 (AP=0.66)，10% 名额下命中率提升至 78%，是随机基线的 2.9 倍
- Cox 模型识别出月付合同的流失风险是两年合同的 14.2 倍 (HR=14.2, 95% CI: 10.1-20.0)
- 最优联系名额为 12%，期望净收益达 4.8 万元

## 方法（简洁版）
...

## 数据限制（一段话，不是多个免责条款）
...
```

---

### 5.3 清理 Git 历史（可选，0.2 天）

**方案：**
- [ ] 创建新分支 `clean-history`
- [ ] 使用 `git filter-branch` 或 `BFG` 移除 `docs/superpowers/` 和已删除的简历素材文件
- [ ] 重写为 10-15 个有意义的提交

**注意：** 会改变提交哈希，需要 force push

---

## 第六阶段：数据升级（可选，需额外时间）

**当前问题：** Telco Churn 是烂大街的数据集

**替代方案（选一个）：**

### 方案 A：KKBox Churn Prediction（推荐）
- Kaggle 竞赛数据，约 200 万条记录，带时间维度
- 可展示时间分割、滑窗特征、时间稳定性评估
- 如果能进 Top 20%，简历效果远超 Telco

### 方案 B：天池/和鲸的流失类竞赛
- 选择仍然开放榜单的比赛
- 可以直接写"天池 XX 赛道 Top 10%"

### 方案 C：生成合成数据
- 使用 CTGAN 或 Copula 生成 10 万条带时间维度的数据
- README 说明是合成数据，避免误导

**时间：** 1-3 天（取决于数据复杂度）

---

## 实施顺序建议

### 最小可行方案（2-3 天，用于快速投递）
1. 第一阶段：修正 Git 问题（0.5 天）
2. 第二阶段 2.1-2.2：特征工程 + LightGBM（1.5 天）
3. 第二阶段 2.3：SHAP 解释（0.5 天）
4. 第五阶段 5.2：重写 README（0.3 天）

**核心产出：** 模型性能提升、有 SHAP、README 清晰

---

### 完整优化方案（4-5 天，用于重点公司）
1. 最小可行方案（2.5 天）
2. 第三阶段：生存分析 + 统计推断（1 天）
3. 第四阶段：业务分析（1 天）
4. 第五阶段 5.1、5.3：AI 降级 + Git 清理（0.5 天）

**核心产出：** 统计专业优势突出、业务思考深入

---

### 竞争力最大化方案（7-10 天，用于梦想公司）
1. 完整优化方案（5 天）
2. 第六阶段：更换数据集并参加比赛（2-5 天）

**核心产出：** "XX 竞赛 Top X%"的简历亮点

---

## 关键指标对比

| 指标 | 当前状态 | 优化后目标 |
|:-----|:---------|:----------|
| 主模型 AP | 0.63 | 0.70-0.72 |
| 特征数 | 19 | 30+ |
| 模型数 | 2（LR、RF） | 3+（LR、RF、LightGBM） |
| 是否调参 | 否 | 是（Optuna） |
| 可解释性 | RF 重要性（错配） | LR 系数 + SHAP |
| 统计推断 | 卡方检验 | 卡方 + 生存分析 + Bootstrap CI |
| 业务分析 | 固定 10% | 敏感性曲线 + 收益模型 |
| 代码行数占比 | 建模 30% / UI 70% | 建模 60% / UI 40% |
| 数据集 | Telco（烂大街） | 可选升级到竞赛数据 |

---

## 技术栈更新

**新增依赖：**
```txt
lightgbm>=4.0
optuna>=3.0
shap>=0.42
lifelines>=0.28
```

**新增文件：**
```
outputs/project_vX/
├── logistic_coefficients.csv       # LR 系数与 OR 值
├── shap_summary.png                # SHAP 汇总图
├── shap_examples/                  # 个案 waterfall
├── survival_curves.png             # KM 曲线
├── cox_results.csv                 # Cox 模型结果
├── budget_sensitivity.csv          # 名额敏感性
├── budget_curve.png                # 名额曲线
└── lightgbm_best_params.json       # 最优超参
```

---

## 风险与注意事项

1. **时间风险：** Optuna 调参可能需要 1-2 小时运行时间，建议先用小规模验证
2. **环境依赖：** LightGBM 在 Windows 上有时需要额外编译工具，建议用 conda 安装
3. **内存占用：** SHAP 计算在大数据集上可能内存不足，可以只对测试集的 1000 条样本计算
4. **过拟合风险：** 特征工程后记得检查验证集表现，避免只在测试集上提升

---

## 下一步行动

**选择执行方案后：**
1. 创建新分支 `git checkout -b feature/optimization`
2. 按阶段逐步实施，每完成一个模块就提交一次
3. 全部完成后合并到 main，打 tag `v2.0-optimized`

**需要我直接实施吗？请选择：**
- 方案 A：最小可行方案（2-3 天工作量）
- 方案 B：完整优化方案（4-5 天工作量）
- 方案 C：你自己按计划实施，我只回答遇到的问题
