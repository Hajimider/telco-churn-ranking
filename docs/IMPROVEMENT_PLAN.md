# 项目审查与修改计划（面向数据挖掘实习投递）

审查对象：`telco-churn-ranking`（电信客户流失预测与排序）。审查日期：2026-11-14。

> 本文为历史审查及建议，不是当前完成清单。后续实现已纠正部分建议：重复CV不能当独立样本；仅凭p值不显著不能强制选LR；静态快照不默认执行生存推断；cross_validate本身不会自动生成OOF；示例简历的模型名和旧107/140不可混用。当前方法见 README、docs/RUNNING.md；验证状态见 telco_churn/IMPLEMENTATION_STATUS.md。
审查方式：通读 `telco_churn/` 全部模块、`sql/`、`tests/` 的用例清单、README，核对 `outputs/run_optimized_v2/` 中的实际结果文件，并对原始数据和预测文件做了几处数值复算。文中数字均来自这些文件。

## 0. 总体判断

**优点（保留）**

- 先切分后分析，预处理放进 Pipeline，没有明显的标签泄漏。
- 统计方法选得对：Wilson 区间、卡方 + Holm 校正、Cramér's V、折外阈值选择，这是应用统计专业的真实优势。
- README 对"关联不等于因果""不是实测收益"的表述克制，诚实。
- 有测试、有看板、有部署，工程完整度高于同类学生项目。

**核心问题（一句话）**

项目"做得很多、证明得很少"：LightGBM 相对逻辑回归的 AP 只高 0.004，却叠加了 Optuna 100 轮、SHAP、生存分析、AI 解读和 755 行看板；而真正能证明水平的几项（特征消融、概率校准、无偏的模型比较、业务假设的敏感性）没有做，或做得不严谨。对"双非 + 实习"的求职者，面试官会抓这些点，而不会数你做了多少模块。

## 1. 问题清单

按严重程度分级：**P0** 会被面试官当场问倒或属于错误；**P1** 明显短板；**P2** 加分项。

### A. 统计与建模的严谨性

| 级别 | 问题 | 证据 |
|:--|:--|:--|
| P0 | **模型比较的显著性结论不成立。** 5 折 CV 得分共用训练数据，彼此不独立，普通配对 t 检验会偏乐观（需 Nadeau-Bengio 修正或重复 CV）。另外三组两两比较没有做多重比较校正，项目里分群检验用了 Holm，这里却没有。 | `model_comparison_test.csv`：逻辑回归 vs 随机森林 p=0.0436 被标为"显著"。按 Holm 校正，三个 p 值 0.0436、0.0615、0.3442 变为 0.131、0.131、0.344，全部不显著。 |
| P0 | **LightGBM 的 CV 分数是乐观的。** Optuna 在 5 折 CV 上搜索 100 轮取最大值，随后又用同一组折（同一个 `random_state=42`）的 AP 去和另外两个未调参模型比较、选主模型。这是选择偏差，不是公平比较。 | `lightgbm_tuning.py` 与 `pipeline.py` 的 `StratifiedKFold` 参数相同。LightGBM 0.6648 对逻辑回归 0.6610，差 0.0038，远小于折间标准差 0.02。 |
| P0 | **"最优名额 40%"是在测试集上选出的。** 净收益曲线用测试集标签算，再取最大值，这等于用测试集调参，而 README 又把测试集称为"固定评估"。 | `pipeline.py` 的 `budget_sensitivity_analysis(y_test, ...)`。 |
| P0 | **"敏感性分析"只让名额 K 变化，没有对业务假设做敏感性。** 触达成本 50、月值 60、12 个月、成功率 30% 都是硬编码单点值，结论完全取决于它们。盈亏平衡精确率 = 50 ÷ (0.3×60×12) = 23.1%，而整体流失率是 26.5%，所以只要假设略变，"联系 40%"就会变成"联系几乎所有人"或"一个也不联系"。 | `budget_analysis.py` 默认参数。 |
| P1 | **LightGBM 概率没有校准，且被 `scale_pos_weight` 扭曲。** 测试集上 LightGBM 平均预测概率 0.411，实际流失率 0.265；逻辑回归 0.267。README 只是声明"不要当概率用"，没有做校准。按概率估算收益、按 0.5 阈值出混淆矩阵都依赖校准。 | `predictions.csv` 复算。 |
| P1 | **特征工程没有证据。** 新增了十余个衍生特征，但没有"有/无衍生特征"的对比。README 自己把消融写成了"后续可补充"。此外 `tenure`、`TotalCharges`、`tenure_contract_month/year`、`charges_deviation` 高度共线，会削弱逻辑回归系数的可解释性。 | 全仓库搜索不到 ablation/消融实现；`feature_importance.csv` 前 6 名全是这类特征。 |
| P1 | **Cox 模型未检验比例风险假设**，且存在完全共线的哑变量。`InternetService_No`、`OnlineSecurity_No internet service`、`TechSupport_No internet service` 三行系数、HR、p 值完全相同，靠 `penalizer=0.1` 勉强拟合；L2 惩罚下的置信区间也不能按常规解读。 | `cox_results.csv` 第 6-8 行；`survival_analysis.py` 没有 `check_assumptions`。 |
| P1 | **只有一次 80/20 划分。** 测试集 1409 人、374 个正例，AP 的 Bootstrap 区间很宽，三个模型的测试集差异（AP 0.638 / 0.647 / 0.659）实际不可区分，但 README 表格以加粗突出 LightGBM。 | `metrics.json`。 |
| P2 | `max_depth` 搜到 3，同时 `num_leaves=38`（深度 3 最多 8 个叶子），搜索空间里两个参数相互冗余。 | `metrics.json` 的 LightGBM 最优参数。 |

### B. 代码质量

| 级别 | 问题 | 位置 |
|:--|:--|:--|
| P0 | **逻辑回归 Bootstrap 会原地修改已训练模型。** `model.__class__(model.steps)` 复用了同一批 estimator 对象而不是克隆，每次 `fit` 都改写原模型；后面还用了裸 `except:` 吞掉所有错误。目前系数在 Bootstrap 之前取出，所以结果碰巧没错，但这是潜在 bug。应改用 `sklearn.base.clone`。 | `model_explanation.py` |
| P1 | `run_pipeline` 一个函数 250 行，包揽切分、训练、检验、导出、画图，难以单测。 | `pipeline.py` |
| P1 | `analyze_segments` 和 `analyze_contract_tenure` 各调用一次 `analyze_training_segments`，每次都重新连 MySQL、建表、上传 5634 行，同一个查询跑了两遍。 | `pipeline.py` |
| P1 | `cross_val_score` 与 `cross_val_predict` 对每个模型各拟合一遍，Optuna 之后又拟合一遍，重复计算；用 `cross_validate` + 一次折外预测即可。 | `pipeline.py` |
| P2 | 随机种子用全局 `np.random.seed`，应改为 `np.random.default_rng(seed)`；`print` 应改为 `logging`；数值型 LightGBM 参数在 `metrics.json` 里被写成 `227.0`；预设常量（名额 10%、K 范围、1000 次 Bootstrap）散落在多处。 | 多处 |
| P2 | 混淆矩阵用 0.5 阈值，但前面刚选出折外阈值，两者在报告中并列，逻辑不一致。 | `pipeline.py` |

### C. 工程与可复现性

| 级别 | 问题 | 说明 |
|:--|:--|:--|
| P0 | **别人无法直接跑通训练和测试。** 训练、SQL 分群、集成测试都强依赖本地 MySQL 8.0 和 `.env`；而 SQL 部分只做了三个 `GROUP BY` 加一个 `RANK()`，pandas 一行就能完成。面试官或同学拉下代码，`pytest` 会直接失败。 | 建议保留 SQL，但提供无服务器后端（见 3.3）。 |
| P0 | **工作区处于半迁移状态。** 当前 `git status` 显示根目录 8 个旧模块被删除、`telco_churn/`、`optional_ai/`、`outputs/run_optimized_v2/` 均未跟踪。GitHub 上的 HEAD 仍是旧结构，README 却描述新结构，别人看到的仓库是对不上的。 | 先提交，见 3.1。 |
| P1 | GitHub 提交作者为 `Pythoner-AI <...@qq.com>`，本机 git 用户是 `Hajimider`，两个身份并存，贡献图可能挂不到你的主页；提交信息有 "上传""优化""renew" 这类无信息量的描述，且暴露了 QQ 邮箱。 | `git log` |
| P1 | 仓库里堆着过期产物和 AI 工具痕迹：`outputs/project_v1`、`project_v2`、只有一张图的 `outputs/run_optimized`、`docs/superpowers/specs/`、一份 365 行的 `docs/OPTIMIZATION_PLAN.md`（README 还链接了它）。招聘方点进去会看到一份写给 AI 的任务清单。 | |
| P1 | `docs/OPTIMIZATION_PLAN.md` 中声称的内容与实现不符：特征消融、时间稳定性、Uplift 讨论、"简化 AI 模块"、README 重写（一句话结论版）、数据升级都没落地；README 仍有 298 行，结论不在开头。 | |
| P2 | 没有 CI，没有固定依赖版本（`requirements.txt` 全是 `>=`），没有 `pyproject.toml`/`Makefile`。 | |
| P2 | 测试 711 行，近一半在测看板和 AI 回退，核心建模逻辑（特征工程、阈值选择之外的部分）覆盖薄；测试里跑完整 `run_pipeline` 依赖 MySQL。 | `tests/test_project.py` |

### D. 与岗位匹配度

| 级别 | 问题 |
|:--|:--|
| P1 | **数据集是最常见的 Kaggle Telco 数据**，几乎所有投递者的简历上都有。7043 行、静态快照、无时间戳，做不出"数据挖掘"岗位看重的特征构造和时间切分。 |
| P1 | **SQL 能力展示太薄。** 数据挖掘/数据分析实习几乎必考 SQL，而项目里的 SQL 只有一个分组 + 一个窗口函数。 |
| P1 | **AI 解读和看板占了篇幅，但不是岗位要求。** README 前半部分和代码量都偏向"展示层"，而特征、验证、业务解读这类核心内容反而被稀释。 |
| P2 | 业务落脚点停在"流失倾向排序"。README 已承认这回答不了"联系是否会改变结局"，但没有给出替代做法（Uplift）。 |

## 2. 修改计划总览

| 阶段 | 目标 | 预计工时 | 投递前必须完成 |
|:--|:--|:--|:--|
| 阶段 1 | 仓库清理与可复现 | 0.5-1 天 | 是 |
| 阶段 2 | 修正统计与代码错误 | 1-1.5 天 | 是 |
| 阶段 3 | 补上有证据的建模结论（消融、校准、阈值与收益） | 1-1.5 天 | 是 |
| 阶段 4 | 数据与业务升级（第二个数据集 / Uplift / 深化 SQL） | 3-5 天 | 否，最能拉开差距 |
| 阶段 5 | README 与简历、面试材料 | 0.5-1 天 | 是 |

时间紧时只做阶段 1、2、5 加阶段 3 的消融与校准，约 3-4 天。

## 3. 分阶段步骤

### 阶段 1：仓库清理与可复现

1. 把当前工作区的重构提交掉，让 GitHub 与 README 一致。先 `git status` 核对，再分两次提交：一次"重构为 telco_churn 包"，一次"新增 run_optimized_v2 报告"。不要用 `git add -A` 一把梭，确认 `.env` 没进暂存区。
2. 统一提交身份：`git config user.name` / `user.email` 设为 GitHub 账号和其 noreply 邮箱，今后的提交信息写成"动词 + 对象"（如"修正逻辑回归 Bootstrap 的模型克隆"）。历史提交不建议改写，仓库已公开且有合并记录，改历史收益小、风险大。
3. 删除或移出仓库：`docs/superpowers/`、`outputs/project_v1`、`outputs/project_v2`、`outputs/run_optimized`；同步修改 `.gitignore` 里对应的白名单，并确认看板默认指向 `run_optimized_v2`。
4. `docs/OPTIMIZATION_PLAN.md` 不要再作为文档链接在 README 里；本文件处理完后，两份计划都不要保留在公开仓库（本地存档即可）。
5. `requirements.txt` 用 `pip freeze` 的结果固定主要依赖版本，新增 GitHub Actions：安装依赖、运行不依赖 MySQL 的测试。

验收：全新环境 `git clone` 后，`pip install -r requirements.txt && pytest` 通过，`streamlit run dashboard.py` 能打开。

### 阶段 2：修正统计与代码错误

1. **模型比较改用无偏流程。**
   - 外层用 `RepeatedStratifiedKFold(5, 3)`，三个模型各自在每个外层训练折内部调参（LightGBM 用较少轮数，如 30，内层 3 折），得到嵌套 CV 的 AP。
   - 差异检验用 Nadeau-Bengio 修正的 t 检验（方差乘以 `1/k + n_test/n_train`），或者直接报告"差值均值 ± 区间"而不下显著性结论。
   - 三组两两比较复用已有的 `holm_adjust`。
2. **名额与阈值在训练集折外预测上确定。** 用 `oof_probabilities` 算净收益曲线，选出 K，再到测试集上报告该 K 的结果；测试集只用于最终汇报，不再参与选 K。
3. **业务假设做真正的敏感性分析。** 对成本 {20, 50, 100}、成功率 {0.1, 0.2, 0.3}、剩余月数 {6, 12, 24} 做网格，输出"各组合下最优 K 与净收益"的热力图或表；在 README 里写盈亏平衡精确率公式和结论的适用范围。客户月值改用每位客户自己的 `MonthlyCharges`，而不是统一的 60。
4. **修正逻辑回归 Bootstrap**：用 `sklearn.base.clone(model)`，去掉裸 `except`，改用 `np.random.default_rng`。同样处理 `bootstrap_metrics`。
5. **Cox 模型**：去掉与 `InternetService` 完全重叠的 "No internet service" 哑变量；调用 `cph.check_assumptions` 并把结果写进报告；对 `Contract` 违反比例风险时，改为按合同类型分层（`strata=["Contract"]`），并把 `penalizer` 降到很小或为 0 以得到可解释的区间。
6. 拆分 `run_pipeline`：至少拆成 `split_data`、`fit_and_score_models`、`evaluate_policies`、`export_report` 四个函数；分群查询只调用一次；用 `cross_validate` 替代重复的 `cross_val_score` + `cross_val_predict`。

验收：`model_comparison_test.csv` 含校正后 p 值；README 不再出现未经校正的"显著"；`pytest` 中新增逻辑回归 Bootstrap 不改变原模型的回归测试。

### 阶段 3：补上有证据的建模结论

1. **特征消融**（最重要的一项）：同一套嵌套 CV 下比较四组特征——原始特征；原始 + 费用类；原始 + 服务组合类；全部衍生特征。每组报告 AP 的均值与标准差，用表格呈现。如果衍生特征没有带来可见提升，就如实写出来，并删掉无用特征。这个结论比"做了 14 个特征"有价值得多。
2. **共线性处理**：对数值特征算 VIF 或相关矩阵，删除或合并 `tenure_contract_month/year`、`TotalCharges` 等；逻辑回归哑变量用 `drop="first"`，并在报告里说明系数的口径（标准化后每 1 个标准差）。
3. **概率校准**：对三个模型画校准曲线，报告 Brier 分数；对 LightGBM 去掉 `scale_pos_weight` 或用 `CalibratedClassifierCV`（isotonic/sigmoid，在训练集内部交叉验证）做校准，比较校准前后的 Brier 与 AP。校准后的概率才能代入收益公式。
4. **更稳的评估**：除一次 80/20 划分外，补重复分层 CV 下的 AP/AUC 区间；PR 曲线加入基线（流失率）。
5. **时间维度的诚实处理**：数据没有时间戳，不要做假的"时间外验证"。在 README 里只写一句话说明，并把它作为选第二个数据集的理由（见阶段 4）。
6. 同时在 `metrics.json` 里记录 `git` 提交号、依赖版本和随机种子，保证结果可追溯。

验收：README 的结果表有"嵌套 CV 均值 ± 标准差"，消融表和校准图都在报告里。

### 阶段 4：数据与业务升级（可选，按投递目标选其一）

**方案 A：用公开随机实验数据做 Uplift（推荐，最贴合应用统计背景）**
- 数据：Criteo Uplift 或 Hillstrom Email Marketing（含随机分组的处理组/对照组）。
- 做法：T-learner / S-learner / 因果森林（`causalml` 或 `econml`），用 Qini 曲线和 AUUC 评估；对比"按流失倾向排序"和"按 uplift 排序"在同一名额下的增量转化。
- 价值：直接回答 README 里承认的短板——"联系是否改变结局"，也能自然引出 A/B 检验、置信区间、样本量计算这些统计强项。

**方案 B：换一个有时间戳的流失数据集**
- 例如 KKBox（WSDM Cup 2018）订阅流失，或天池/和鲸上的用户行为数据。
- 做法：用 SQL 或 pandas 从日志构造 RFM、近 7/30 天行为次数、趋势类特征，按时间切分训练和验证，做 PSI 稳定性检验。
- 价值：特征工程和数据处理量远大于 Telco，也更接近真实实习任务。

**方案 C：深化 SQL（任何方案都建议做）**
- 把 SQL 做成一份独立的 `sql/` 脚本集，内容覆盖：留存/同期群（cohort）分析、`LAG/LEAD` 求相邻两次行为间隔、`ROW_NUMBER` 取每个用户的最近一次记录、`SUM() OVER` 做累计与滑动窗口、`CASE WHEN` 的分群漏斗。每段附一句业务问题和结果截图。
- 后端增加 DuckDB 或 SQLite 选项，无 MySQL 时自动回退；MySQL 路径保留，作为"可选"项。

### 阶段 5：README、简历与面试材料

1. **README 重写到 120 行以内。** 顺序：一句话结论（含数值和适用边界）→ 三张图（模型对比、Lift 曲线、消融表）→ 一屏运行说明 → 方法要点 → 局限。AI 解读、MySQL 配置、PyCharm 步骤、Streamlit 部署移到 `docs/` 下的附录。
2. 看板和 AI 模块保留但降级：README 只用一小节介绍，并标注为"可选展示层"。不要在简历首行写 AI 解读。
3. 简历写法（数字待阶段 2、3 的新结果替换）：
   - 反例：「LightGBM + Optuna 调参，AP 提升显著」。
   - 正例：「在 7043 条电信客户数据上构建流失排序流程；以嵌套 5 折 CV 比较逻辑回归/随机森林/LightGBM，三者 AP 差异在 Holm 校正后不显著，据此选用可解释的逻辑回归作为基线；联系前 10% 客户命中流失 107/140（Lift 2.88）；特征消融显示 X 类特征带来 +0.0XX AP」。
   - 突出的是"检验后得出的结论"，而不是堆技术名词。
4. 准备一页面试问答（放本地，不进仓库），至少写清楚：
   - 为什么 LightGBM 只比逻辑回归好 0.004 还要用它？（答：没有显著差异，所以报告里没把它说成提升）
   - 为什么用 AP 而不是 AUC 或 Accuracy？
   - 为什么折外阈值，而不是测试集上选阈值？
   - `scale_pos_weight` 对概率有什么影响，怎么校准？
   - Wilson 区间和 Wald 区间的区别；Holm 和 Bonferroni 的区别；Cramér's V 的意义。
   - Cox 模型里 tenure 是什么，删失是什么，比例风险假设是什么。
   - 盈亏平衡精确率怎么算，成功率变了结论怎么变。
   - 代码里每个文件为什么存在，哪些是你自己写的、哪些借助了工具。这类问题要能逐行讲清楚。

## 4. 建议的执行顺序

| 顺序 | 任务 | 对应问题 |
|:--|:--|:--|
| 1 | 提交重构、清理仓库、统一提交身份 | C 的 P0、P1 |
| 2 | 修逻辑回归 Bootstrap、拆函数、去重复计算 | B |
| 3 | 嵌套 CV + Nadeau-Bengio + Holm | A 的 P0 |
| 4 | OOF 上选 K，业务假设网格敏感性 | A 的 P0 |
| 5 | 特征消融 + 共线性 + 校准 | A 的 P1 |
| 6 | Cox 假设检验与分层 | A 的 P1 |
| 7 | 无 MySQL 的后端、CI、依赖固定 | C 的 P0、P2 |
| 8 | README 重写、简历与面试材料 | 阶段 5 |
| 9 | Uplift 或第二个数据集 + 深化 SQL | D |

## 5. 不建议做的事

- 不要继续往看板和 AI 模块上加功能，这不是岗位考察点。
- 不要为了让 LightGBM"赢"而继续调参或换随机种子；如果嵌套 CV 下它没有优势，就把逻辑回归当主模型，并把这一点写成亮点。
- 不要在简历里写没有做过的内容（时间外验证、因果效应、实测收益）。
- 不要改写已公开的 Git 历史。
- 不要合成数据凑"大规模"。
