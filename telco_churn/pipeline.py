"""客户分群、流失模型训练和固定名额下的排序评估。"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import chi2_contingency

from .chart_font import configure_chart_font
from .lightgbm_tuning import tune_lightgbm
from .model_explanation import explain_logistic_regression, explain_shap
from .statistical_inference import bootstrap_metrics
from .budget_analysis import budget_sensitivity_analysis
from .mysql_segments import analyze_training_segments

configure_chart_font()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "run_validated"
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
REQUIRED_COLUMNS = {
    "customerID", "Churn", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
}
CATEGORICAL_COLUMNS = sorted(REQUIRED_COLUMNS - {"customerID", "Churn"} - set(NUMERIC_COLUMNS))

# 特征工程后的数值列和类别列
FEATURE_GROUPS = {
    "原始特征": NUMERIC_COLUMNS,
    "原始+费用": NUMERIC_COLUMNS + ["charges_deviation", "has_high_charges"],
    "原始+服务": NUMERIC_COLUMNS + ["service_count", "has_internet", "full_protection"],
    "全部衍生": NUMERIC_COLUMNS + [
        "charges_deviation", "has_high_charges", "service_count", "has_internet",
        "full_protection", "is_new_customer", "is_loyal_customer", "tenure_contract_month",
        "auto_payment", "paperless_risky", "has_family", "single_senior",
    ],
}
# tenure_contract_year 与 tenure - tenure_contract_month 恒等，不重复进入模型。
ENGINEERED_NUMERIC = FEATURE_GROUPS["全部衍生"]
ENGINEERED_CATEGORICAL = CATEGORICAL_COLUMNS  # 类别列保持不变
AI_CONTEXT_COLUMNS = [
    "tenure", "Contract", "MonthlyCharges", "InternetService", "PaymentMethod",
    "TechSupport", "OnlineSecurity",
]


def prepare_data(frame):
    """检查原始表并把标签、编号和建模特征分开。"""
    if frame.empty:
        raise ValueError("CSV 中没有客户记录。")
    if frame.columns.duplicated().any():
        raise ValueError("CSV 中存在重复列名，请检查文件表头。")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要字段：{', '.join(sorted(missing))}")

    ids = frame["customerID"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("customerID 不能为空且必须唯一。")
    labels = frame["Churn"].astype("string").str.strip().str.lower().map({"yes": 1, "no": 0})
    if labels.isna().any() or labels.nunique() != 2:
        raise ValueError("Churn 必须同时包含 Yes 和 No，不能有其他标签。")

    X = frame.drop(columns=["customerID", "Churn"]).copy()
    for column in NUMERIC_COLUMNS:
        raw = X[column].astype("string").str.strip()
        values = pd.to_numeric(raw.replace("", pd.NA), errors="coerce")
        invalid = raw.notna() & raw.ne("") & values.isna()
        if invalid.any():
            raise ValueError(f"{column} 列含有无法转换的数值。")
        X[column] = values.astype(float)
        if np.isinf(X[column]).any() or (X[column].dropna() < 0).any():
            raise ValueError(f"{column} 须为非负有限数值或缺失值。")
    if X[["tenure", "MonthlyCharges"]].isna().any().any():
        raise ValueError("tenure 和 MonthlyCharges 不能缺失（用于排序、分群及情景分析）。")
    if not X["SeniorCitizen"].dropna().isin([0, 1]).all():
        raise ValueError("SeniorCitizen 须为 0 或 1。")
    return X, labels.astype(int), ids


def engineer_features(X):
    """创建衍生特征以增强模型表现。"""
    X = X.copy()

    # 费用相关特征
    X["charges_deviation"] = np.where(
        X["tenure"] > 0,
        X["TotalCharges"] / X["tenure"] - X["MonthlyCharges"],
        0
    )
    X["has_high_charges"] = (X["MonthlyCharges"] >= 70).astype(int)

    # 服务组合特征
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    X["service_count"] = sum((X[col] == "Yes").astype(int) for col in service_cols)
    X["has_internet"] = (X["InternetService"] != "No").astype(int)

    protection_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    X["full_protection"] = (sum((X[col] == "Yes").astype(int) for col in protection_cols) == 4).astype(int)

    # 客户生命周期特征
    X["is_new_customer"] = (X["tenure"] <= 6).astype(int)
    X["is_loyal_customer"] = (X["tenure"] >= 36).astype(int)
    X["tenure_contract_month"] = (X["Contract"] == "Month-to-month").astype(int) * X["tenure"]
    X["tenure_contract_year"] = (X["Contract"].isin(["One year", "Two year"])).astype(int) * X["tenure"]

    # 支付与账单特征
    X["auto_payment"] = (X["PaymentMethod"].isin(["Bank transfer (automatic)",
                                                    "Credit card (automatic)"])).astype(int)
    X["paperless_risky"] = ((X["PaperlessBilling"] == "Yes") &
                            (X["PaymentMethod"] == "Electronic check")).astype(int)

    # 家庭客户特征
    X["has_family"] = ((X["Partner"] == "Yes") | (X["Dependents"] == "Yes")).astype(int)
    X["single_senior"] = ((X["SeniorCitizen"] == 1) &
                          (X["Partner"] == "No") &
                          (X["Dependents"] == "No")).astype(int)

    return X


def make_pipeline(estimator, numeric_cols=None, categorical_cols=None):
    """预处理随模型一起拟合，避免交叉验证时泄漏验证折信息。"""
    if numeric_cols is None:
        numeric_cols = ENGINEERED_NUMERIC
    if categorical_cols is None:
        categorical_cols = ENGINEERED_CATEGORICAL

    numeric = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, numeric_cols),
        ("categorical", categorical, categorical_cols),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def prepare_output(path):
    """只允许写入本项目目录下的空输出目录。"""
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("输出目录必须位于本项目目录内。")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("输出目录非空；请换一个新目录，例如 outputs/run_02。")
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze_segments(training_frame):
    """用 MySQL 对训练集做客户分群，只输出描述性统计。"""
    overall, segments, _ = analyze_training_segments(training_frame)
    segments["小样本"] = segments["客户数"] < 30
    return overall, segments


def analyze_contract_tenure(training_frame):
    """用 MySQL 汇总合同与在网时长交叉分群。"""
    _, _, cross_segments = analyze_training_segments(training_frame)
    return add_wilson_intervals(cross_segments)


def wilson_interval(successes, total, z=1.959963984540054):
    """返回二项比例的 Wilson 95% 区间。"""
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def add_wilson_intervals(frame):
    frame = frame.copy()
    intervals = [wilson_interval(int(row["流失人数"]), int(row["客户数"]))
                 for _, row in frame.iterrows()]
    frame["流失率下限"] = [interval[0] for interval in intervals]
    frame["流失率上限"] = [interval[1] for interval in intervals]
    return frame


def holm_adjust(p_values):
    """对有效 p 值执行 Holm 逐步校正，保留输入顺序。"""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(order) - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def analyze_segment_associations(training_frame, dimensions):
    """按训练集分群维度检验与流失标签的关联，并对有效检验做 Holm 校正。"""
    rows = []
    for dimension, column in dimensions.items():
        table = pd.crosstab(training_frame[column], training_frame["Churn"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            rows.append({"分群维度": dimension, "卡方统计量": np.nan, "自由度": 0,
                         "p值": np.nan, "校正后p值": np.nan, "Cramér's V": np.nan,
                         "检验状态": "类别不足，未检验"})
            continue
        statistic, p_value, degrees, expected = chi2_contingency(table, correction=False)
        if (expected < 5).any():
            rows.append({"分群维度": dimension, "卡方统计量": np.nan, "自由度": int(degrees),
                         "p值": np.nan, "校正后p值": np.nan, "Cramér's V": np.nan,
                         "检验状态": "期望频数小于5，未报告检验"})
            continue
        total = table.to_numpy().sum()
        denominator = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = np.sqrt(statistic / (total * denominator)) if denominator else np.nan
        rows.append({"分群维度": dimension, "卡方统计量": float(statistic), "自由度": int(degrees),
                     "p值": float(p_value), "校正后p值": np.nan,
                     "Cramér's V": float(cramers_v), "检验状态": "已检验"})
    adjusted = holm_adjust([row["p值"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["校正后p值"] = float(value) if np.isfinite(value) else np.nan
    return pd.DataFrame(rows)


def select_f1_threshold(labels, probabilities):
    """按 F1 最大选择阈值；并列时先选最接近 0.5，再选较高阈值。"""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("阈值选择需要等长非空标签与概率。")
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("阈值选择需要两类二值标签。")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("概率必须有限且位于 [0,1]。")
    precision, recall, candidates = precision_recall_curve(labels, probabilities)
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                       out=np.zeros_like(denominator), where=denominator != 0)
    extras = np.array([0.0, 0.5, 1.0])
    extra_scores = [f1_score(labels, probabilities >= value, zero_division=0) for value in extras]
    candidates = np.r_[candidates, extras]
    scores = np.r_[scores, extra_scores]
    tied = candidates[np.isclose(scores, scores.max())]
    return float(min(tied, key=lambda threshold: (abs(threshold - 0.5), -threshold)))


def evaluate_policy(name, labels, ids, scores, budget_count):
    """按分数和客户编号稳定排序，评估固定容量下的离线命中。"""
    labels = np.asarray(labels, dtype=int)
    ids = np.asarray(ids, dtype=str)
    scores = np.asarray(scores, dtype=float)
    order = np.lexsort((ids, -scores))
    selected = order[:budget_count]
    positives = int(labels.sum())
    hits = int(labels[selected].sum())
    precision = hits / budget_count if budget_count else 0.0
    recall = hits / positives if positives else 0.0
    prevalence = positives / len(labels) if len(labels) else 0.0
    lift = precision / prevalence if prevalence else 0.0
    return {
        "策略": name,
        "名额": int(budget_count),
        "命中流失": hits,
        "Precision@K": precision,
        "Recall@K": recall,
        "Lift@K": lift,
    }, order


def _tenure_group(value):
    if value <= 12:
        return "1-12 个月"
    if value <= 24:
        return "13-24 个月"
    if value <= 48:
        return "25-48 个月"
    return "49 个月及以上"


def _monthly_group(value):
    if value < 35:
        return "低于 35"
    if value < 70:
        return "35-69.99"
    return "70 及以上"


def build_ai_context(X_test, test_ids, selected_model, scores, order, budget_count, segments, overall):
    """导出主模型入选客户的资料，不包含测试集标签。"""
    rate_maps = {
        dimension: group.set_index("分群")["流失率"].to_dict()
        for dimension, group in segments.groupby("分群维度")
    }
    overall_rate = float(overall.iloc[0]["流失率"])
    rows = []
    for rank, position in enumerate(order[:budget_count], start=1):
        values = X_test.iloc[position]
        tenure = float(values["tenure"])
        monthly_charges = float(values["MonthlyCharges"])
        contract = str(values["Contract"])
        contract_rate = float(rate_maps.get("合同类型", {}).get(contract, overall_rate))
        tenure_rate = float(rate_maps.get("在网时长", {}).get(_tenure_group(tenure), overall_rate))
        monthly_rate = float(rate_maps.get("月费区间", {}).get(
            _monthly_group(monthly_charges), overall_rate))
        score = float(scores[position])
        evidence = []
        if score >= 0.7:
            evidence.append("主模型排序分数较高")
        if contract == "Month-to-month":
            evidence.append("月付合同")
        if tenure <= 12:
            evidence.append("在网时间较短")
        if monthly_charges >= 70:
            evidence.append("月费处于较高区间")
        if contract_rate > overall_rate:
            evidence.append(f"合同类型历史流失率较高（{contract_rate:.1%}）")
        if tenure_rate > overall_rate:
            evidence.append(f"在网时长分组历史流失率较高（{tenure_rate:.1%}）")
        if monthly_rate > overall_rate:
            evidence.append(f"月费分组历史流失率较高（{monthly_rate:.1%}）")
        rows.append({
            "customerID": str(test_ids.iloc[position]),
            "model_name": selected_model,
            "rank": rank,
            "risk_score": score,
            **{column: values[column] for column in AI_CONTEXT_COLUMNS},
            "contract_churn_rate": contract_rate,
            "tenure_group_churn_rate": tenure_rate,
            "monthly_group_churn_rate": monthly_rate,
            "risk_evidence": "；".join(evidence) or "进入主模型固定容量优先名单",
        })
    return pd.DataFrame(rows, columns=[
        "customerID", "model_name", "rank", "risk_score", *AI_CONTEXT_COLUMNS,
        "contract_churn_rate", "tenure_group_churn_rate",
        "monthly_group_churn_rate", "risk_evidence",
    ])


def run_pipeline(frame, output_dir, test_size=0.2, random_state=42, *,
                 outer_splits=5, repeats=3, inner_splits=3, trials=30,
                 n_bootstrap=1000, backend="sqlite", explanations=True):
    """嵌套重复验证、训练折外选阈值与名额、最后评估固定测试集。"""
    from .validation import nested_validation, fit_pair, MODEL_NAMES
    from .budget_analysis import budget_table, select_budget, business_scenarios
    from .reporting import export_diagnostics, run_metadata
    if not isinstance(n_bootstrap, (int, np.integer)) or isinstance(n_bootstrap, bool) or n_bootstrap < 2:
        raise ValueError("Bootstrap 次数必须是至少为2的整数。")
    if backend not in ("sqlite", "mysql"):
        raise ValueError("分群后端须为 sqlite 或 mysql。")
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("测试集比例需在 0.1 到 0.5 之间。")
    X, y, ids = prepare_data(frame)

    # 应用特征工程
    X = engineer_features(X)

    if len(frame) < 50 or y.value_counts().min() < 5:
        raise ValueError("样本量不足：至少需要 50 行，并且每类至少有 5 行以支持五折分层验证。")

    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    if y_test.nunique() != 2 or y_train.nunique() != 2:
        raise ValueError("训练集与测试集均须包含两类标签。")
    output = prepare_output(output_dir)
    # 数值清洗后的训练数据作为分群唯一输入，避免原始字符串参与比较。
    train_frame = X.iloc[train_idx].copy()
    train_frame["customerID"] = ids.iloc[train_idx].to_numpy()
    train_frame["Churn"] = y_train.to_numpy()
    overall, segments, cross_segments = analyze_training_segments(train_frame, backend=backend)
    segments["小样本"] = segments["客户数"] < 30
    segments = add_wilson_intervals(segments)
    inference = analyze_segment_associations(train_frame.assign(
        在网时长分组=train_frame["tenure"].map(_tenure_group),
        月费分组=train_frame["MonthlyCharges"].map(_monthly_group),
    ), {
        "合同类型": "Contract", "在网时长": "在网时长分组", "月费": "月费分组",
        "付款方式": "PaymentMethod", "互联网服务": "InternetService",
        "技术支持": "TechSupport",
    })
    cross_segments = add_wilson_intervals(cross_segments)
    test_ids = ids.iloc[test_idx]
    validation = nested_validation(
        X_train, y_train, FEATURE_GROUPS, outer_splits=outer_splits, repeats=repeats,
        inner_splits=inner_splits, trials=trials, seed=random_state)
    cv_scores = validation["scores"]
    oof_probabilities = validation["oof"]
    models, calibrated_models = {}, {}
    for name in MODEL_NAMES:
        models[name], calibrated_models[name] = fit_pair(
            name, X_train, y_train, ENGINEERED_NUMERIC, random_state, trials, inner_splits)
    best_lgb_params = models["LightGBM"].named_steps["model"].get_params()
    thresholds = {name: select_f1_threshold(y_train, scores)
                  for name, scores in oof_probabilities.items()}
    selected_model = max(models, key=lambda name: float(np.mean(cv_scores[name])))
    predictions = pd.DataFrame({"customerID": test_ids.to_numpy(), "actual_churn": y_test.to_numpy()})
    predictions["tenure"] = X_test["tenure"].to_numpy()
    metrics = {
        "样本数": int(len(frame)),
        "训练集样本数": int(len(train_idx)),
        "测试集样本数": int(len(test_idx)),
        "流失率": float(y.mean()),
        "random_state": int(random_state),
        "test_size": float(test_size),
        "名额比例": 0.10,
        "LightGBM最优参数": best_lgb_params,
        "报告版本": 3,
        "运行溯源": run_metadata(frame, random_state),
        "交叉验证": {
            "折数": outer_splits,
            "重复次数": repeats,
            "内层折数": inner_splits,
            "Optuna轮数": trials,
            "方法": "嵌套重复分层CV，训练折内调参与sigmoid校准",
            "AUC均值": {name: float(np.mean(values)) for name, values in validation["auc_scores"].items()},
            "选择模型": selected_model,
            "AP均值": {name: float(np.mean(scores)) for name, scores in cv_scores.items()},
            "AP标准差": {name: float(np.std(scores, ddof=1)) for name, scores in cv_scores.items()},
            "各折AP": {name: [float(value) for value in scores] for name, scores in cv_scores.items()},
        },
        "模型": {},
        "阈值评估": {name: {"折外F1阈值": threshold,
                           "折外F1": float(f1_score(y_train, oof_probabilities[name] >= threshold,
                                                    zero_division=0))}
                    for name, threshold in thresholds.items()},
    }

    model_scores = {}
    for name, model in calibrated_models.items():
        probability = model.predict_proba(X_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        threshold_predicted = (probability >= thresholds[name]).astype(int)
        model_scores[name] = probability
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_prediction"] = predicted
        metrics["模型"][name] = {
            "Accuracy（准确率）": float(accuracy_score(y_test, predicted)),
            "Precision（查准率）": float(precision_score(y_test, predicted, zero_division=0)),
            "Recall（查全率）": float(recall_score(y_test, predicted, zero_division=0)),
            "F1": float(f1_score(y_test, predicted, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probability)),
            "AP（平均精确率）": float(average_precision_score(y_test, probability)),
            "折外F1阈值": float(thresholds[name]),
            "阈值测试集查准率": float(precision_score(y_test, threshold_predicted, zero_division=0)),
            "阈值测试集查全率": float(recall_score(y_test, threshold_predicted, zero_division=0)),
            "阈值测试集F1": float(f1_score(y_test, threshold_predicted, zero_division=0)),
            "Brier": float(brier_score_loss(y_test, probability)),
        }

    # Bootstrap 置信区间
    logging.info("正在计算 Bootstrap 置信区间...")
    for name, probability in model_scores.items():
        boot_ci = bootstrap_metrics(y_test.to_numpy(), probability, n_bootstrap=n_bootstrap, random_state=random_state)
        metrics["模型"][name]["Bootstrap置信区间"] = boot_ci

    # 模型比较统计检验
    model_comparisons = validation["comparisons"]
    metrics["模型比较检验"] = model_comparisons
    metrics["生存分析"] = {
        "状态": "默认禁用",
        "原因": "静态快照缺少可靠事件日期、入组及删失机制，不能验证标准生存分析假设。",
    }
    validation["folds"].to_csv(output / "nested_cv_folds.csv", index=False, encoding="utf-8-sig")
    validation["ablation"].to_csv(output / "ablation_folds.csv", index=False, encoding="utf-8-sig")
    ablation = validation["ablation"].groupby("特征组")["AP"].agg(["mean", "std"]).reset_index()
    ablation.to_csv(output / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    X_train[ENGINEERED_NUMERIC].corr().to_csv(output / "numeric_correlations.csv", encoding="utf-8-sig")
    pd.DataFrame({"customerID": ids.iloc[train_idx].to_numpy(), "actual_churn": y_train.to_numpy(),
                  **{f"{name}_probability": p for name, p in oof_probabilities.items()}}).to_csv(
                      output / "training_oof.csv", index=False, encoding="utf-8-sig")
    export_diagnostics(validation, X_test, y_test, models, calibrated_models, output)

    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output / "overall_summary.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(output / "segment_summary.csv", index=False, encoding="utf-8-sig")
    inference.to_csv(output / "segment_inference.csv", index=False, encoding="utf-8-sig")
    cross_segments.to_csv(output / "contract_tenure_segments.csv", index=False, encoding="utf-8-sig")
    budget_count = int(len(y_test) * metrics["名额比例"])
    selected_probability = model_scores[selected_model]
    selected_order = np.lexsort((test_ids.to_numpy(dtype=str), -selected_probability))
    ai_context = build_ai_context(
        X_test, test_ids, selected_model, selected_probability, selected_order,
        budget_count, segments, overall)
    ai_context.to_csv(output / "ai_customer_context.csv", index=False, encoding="utf-8-sig")
    metrics["AI上下文"] = {
        "主模型": selected_model,
        "优先客户数": int(len(ai_context)),
        "字段": list(ai_context.columns),
        "不含真实流失标签": True,
    }
    policy_rows = []
    assignment_rows = []
    # 规则和模型都按同一名额排序，分数越大越优先。
    policy_scores = {**model_scores, "短在网时长规则": -X_test["tenure"].to_numpy(dtype=float)}
    for name, scores in policy_scores.items():
        row, order = evaluate_policy(name, y_test, test_ids, scores, budget_count)
        policy_rows.append(row)
        for rank, position in enumerate(order[:budget_count], start=1):
            assignment_rows.append({
                "策略": name,
                "customerID": str(test_ids.iloc[position]),
                "actual_churn": int(y_test.iloc[position]),
                "排名": rank,
                "排序分数": float(scores[position]),
            })

    positives = int(y_test.sum())
    prevalence = positives / len(y_test)
    policy_rows.append({
        "策略": "随机触达期望",
        "名额": budget_count,
        "命中流失": budget_count * prevalence,
        "Precision@K": prevalence,
        "Recall@K": budget_count / len(y_test),
        "Lift@K": 1.0 if positives else 0.0,
    })
    pd.DataFrame(policy_rows).to_csv(output / "policy_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(assignment_rows, columns=[
        "策略", "customerID", "actual_churn", "排名", "排序分数"
    ]).to_csv(output / "policy_assignments.csv", index=False, encoding="utf-8-sig")

    # 名额敏感性分析
    logging.info("正在进行名额敏感性分析...")
    train_budget = budget_table(y_train, oof_probabilities[selected_model], ids.iloc[train_idx],
                                X_train["MonthlyCharges"])
    optimal_k = select_budget(train_budget)
    train_budget.to_csv(output / "budget_training_oof.csv", index=False, encoding="utf-8-sig")
    business_scenarios(y_train, oof_probabilities[selected_model], ids.iloc[train_idx],
                       X_train["MonthlyCharges"]).to_csv(
                           output / "business_scenarios.csv", index=False, encoding="utf-8-sig")
    sensitivity_df, _ = budget_sensitivity_analysis(
        y_test, selected_probability, test_ids, output,
        monthly_values=X_test["MonthlyCharges"], selected_k=optimal_k)
    frozen = sensitivity_df.loc[np.isclose(sensitivity_df["K比例"], optimal_k)].iloc[0]
    metrics["名额敏感性"] = {
        "训练OOF选择名额比例": optimal_k,
        "冻结名额测试集情景净收益": float(frozen["期望净收益"]),
        "选择来源": "仅训练集嵌套折外预测，测试集不选K",
        "月值口径": "客户MonthlyCharges收入代理，非利润/CLV",
        "假设": {"触达成本": 50, "剩余月数": 12, "成功率": 0.3},
        "详见": "budget_sensitivity.csv"
    }
    pd.DataFrame(model_comparisons).to_csv(
        output / "model_comparison_test.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, name in zip(np.atleast_1d(axes), models):
        predicted = (model_scores[name] >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay(confusion_matrix(y_test, predicted, labels=[0, 1]), display_labels=["未流失", "已流失"]).plot(
            ax=ax, colorbar=False, values_format="d")
        ax.set_title(name)
    fig.suptitle("测试集混淆矩阵（训练OOF冻结阈值）")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probability in model_scores.items():
        RocCurveDisplay.from_predictions(y_test, probability, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="随机排序")
    ax.set_title("测试集 ROC 曲线")
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    forest = models["随机森林"]
    names = forest.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "importance": forest.named_steps["model"].feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(
        output / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # 模型解释：逻辑回归系数和 SHAP
    logging.info("正在生成模型解释...")
    lr_model = models["逻辑回归"]
    lr_feature_names = lr_model.named_steps["preprocess"].get_feature_names_out()
    if explanations:
        lr_coef_df = explain_logistic_regression(lr_model, X_train, y_train, lr_feature_names,
                                                 n_bootstrap=n_bootstrap, random_state=random_state)
        lr_coef_df.to_csv(output / "logistic_coefficients.csv", index=False, encoding="utf-8-sig")
        tree_name = "LightGBM" if selected_model == "LightGBM" else "随机森林"
        shap_model = models[tree_name]
        metrics["SHAP模型"] = tree_name + "（未校准基模型，不解释校准映射）"
        explain_shap(shap_model, X_test, shap_model.named_steps["preprocess"].get_feature_names_out(), output)
    else:
        metrics["模型解释"] = "本次配置跳过，非运行失败"

    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="电信客户 CSV 文件")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="项目报告目录，必须为空")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集占比，默认 0.2")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--backend", choices=["sqlite", "mysql"], default="sqlite")
    parser.add_argument("--no-explanations", action="store_true", help="跳过SHAP和LR系数重采样")
    args = parser.parse_args()
    try:
        if not args.data.is_file():
            raise FileNotFoundError(f"找不到数据文件：{args.data}。请查看 data/README.md。")
        metrics = run_pipeline(pd.read_csv(args.data), args.output, args.test_size, args.random_state,
                               outer_splits=args.outer_splits, repeats=args.repeats,
                               inner_splits=args.inner_splits, trials=args.trials,
                               n_bootstrap=args.bootstrap, backend=args.backend,
                               explanations=not args.no_explanations)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f"完成：{metrics['样本数']:,} 行，训练 {metrics['训练集样本数']:,} 行，测试 {metrics['测试集样本数']:,} 行。")
    print(f"训练集嵌套验证选择：{metrics['交叉验证']['选择模型']}。")
    for name, values in metrics["模型"].items():
        print(f"{name}: AP={values['AP（平均精确率）']:.3f}, F1={values['F1']:.3f}, ROC-AUC={values['ROC-AUC']:.3f}")
    print(f"固定 10% 名额策略结果已保存到：{args.output / 'policy_metrics.csv'}")
    print(f"结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
