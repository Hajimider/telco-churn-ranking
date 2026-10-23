"""客户流失分析流程：数据校验、客户分群、模型验证与容量策略评估。"""
import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
for FONT_NAME in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if FONT_NAME in AVAILABLE_FONTS:
        plt.rcParams["font.sans-serif"] = [FONT_NAME]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "project_v1"
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
REQUIRED_COLUMNS = {
    "customerID", "Churn", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
}
CATEGORICAL_COLUMNS = [column for column in sorted(REQUIRED_COLUMNS - {"customerID", "Churn"})
                       if column not in NUMERIC_COLUMNS]


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
    return X, labels.astype(int), ids


def make_pipeline(estimator):
    """把填补、编码和模型串起来，确保预处理只从训练集学习。"""
    numeric = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
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
    """用 SQLite 对训练集做客户分群，只输出描述性统计。"""
    segment_sql = (ROOT / "sql" / "segment_analysis.sql").read_text(encoding="utf-8")
    with sqlite3.connect(":memory:") as connection:
        training_frame[["Contract", "tenure", "MonthlyCharges", "Churn"]].to_sql(
            "training_customers", connection, index=False, if_exists="replace")
        connection.executescript(segment_sql)
        overall = pd.read_sql_query("SELECT * FROM overall_churn", connection)
        segments = pd.read_sql_query("SELECT * FROM customer_segments", connection)
    segments["小样本"] = segments["客户数"] < 30
    return overall, segments


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


def run_pipeline(frame, output_dir, test_size=0.2, random_state=42):
    """交叉验证候选模型，并在固定测试集评估模型和触达排序。"""
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("测试集比例需在 0.1 到 0.5 之间。")
    X, y, ids = prepare_data(frame)
    if len(frame) < 50 or y.value_counts().min() < 5:
        raise ValueError("样本量不足：至少需要 50 行，并且每类至少有 5 行以支持五折分层验证。")

    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_frame = frame.iloc[train_idx].copy()
    train_frame["Churn"] = y_train.to_numpy()
    overall, segments = analyze_segments(train_frame)
    test_ids = ids.iloc[test_idx]

    models = {
        "逻辑回归": make_pipeline(LogisticRegression(max_iter=1000, random_state=random_state)),
        "随机森林": make_pipeline(RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, random_state=random_state, n_jobs=-1)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = {
        name: cross_val_score(model, X_train, y_train, scoring="average_precision", cv=cv, n_jobs=1)
        for name, model in models.items()
    }
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
        "交叉验证": {
            "折数": 5,
            "选择模型": selected_model,
            "AP均值": {name: float(np.mean(scores)) for name, scores in cv_scores.items()},
            "AP标准差": {name: float(np.std(scores, ddof=1)) for name, scores in cv_scores.items()},
            "各折AP": {name: [float(value) for value in scores] for name, scores in cv_scores.items()},
        },
        "模型": {},
    }

    fitted = {}
    for key, model in models.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        fitted[key] = (model, probability, predicted)
        predictions[f"{key}_probability"] = probability
        predictions[f"{key}_prediction"] = predicted
        metrics["模型"][key] = {
            "Accuracy（准确率）": float(accuracy_score(y_test, predicted)),
            "Precision（查准率）": float(precision_score(y_test, predicted, zero_division=0)),
            "Recall（查全率）": float(recall_score(y_test, predicted, zero_division=0)),
            "F1": float(f1_score(y_test, predicted, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probability)),
            "AP（平均精确率）": float(average_precision_score(y_test, probability)),
        }

    output = prepare_output(output_dir)
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output / "overall_summary.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(output / "segment_summary.csv", index=False, encoding="utf-8-sig")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    budget_count = int(len(y_test) * metrics["名额比例"])
    policy_rows = []
    assignment_rows = []
    for model_name, (_, probability, _) in fitted.items():
        row, order = evaluate_policy(model_name, y_test, test_ids, probability, budget_count)
        policy_rows.append(row)
        for rank, position in enumerate(order[:budget_count], start=1):
            assignment_rows.append({
                "策略": model_name,
                "customerID": str(test_ids.iloc[position]),
                "actual_churn": int(y_test.iloc[position]),
                "排名": rank,
                "排序分数": float(probability[position]),
            })

    tenure_scores = -X_test["tenure"].to_numpy(dtype=float)
    row, order = evaluate_policy("短在网时长规则", y_test, test_ids, tenure_scores, budget_count)
    policy_rows.append(row)
    for rank, position in enumerate(order[:budget_count], start=1):
        assignment_rows.append({
            "策略": "短在网时长规则",
            "customerID": str(test_ids.iloc[position]),
            "actual_churn": int(y_test.iloc[position]),
            "排名": rank,
            "排序分数": float(tenure_scores[position]),
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, (_, _, predicted)) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_test, predicted), display_labels=["未流失", "已流失"]).plot(
            ax=ax, colorbar=False, values_format="d")
        ax.set_title(name)
    fig.suptitle("测试集混淆矩阵（默认分类阈值 0.5）")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (_, probability, _) in fitted.items():
        RocCurveDisplay.from_predictions(y_test, probability, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="随机排序")
    ax.set_title("测试集 ROC 曲线")
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    forest = fitted["随机森林"][0]
    names = forest.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "importance": forest.named_steps["model"].feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(
        output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="电信客户 CSV 文件")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="项目报告目录，必须为空")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集占比，默认 0.2")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()
    try:
        if not args.data.is_file():
            raise FileNotFoundError(f"找不到数据文件：{args.data}。请查看 data/README.md。")
        metrics = run_pipeline(pd.read_csv(args.data), args.output, args.test_size, args.random_state)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f"完成：{metrics['样本数']:,} 行，训练 {metrics['训练集样本数']:,} 行，测试 {metrics['测试集样本数']:,} 行。")
    print(f"训练集五折交叉验证选择：{metrics['交叉验证']['选择模型']}。")
    for name, values in metrics["模型"].items():
        print(f"{name}: AP={values['AP（平均精确率）']:.3f}, F1={values['F1']:.3f}, ROC-AUC={values['ROC-AUC']:.3f}")
    print(f"固定 10% 名额策略结果已保存到：{args.output / 'policy_metrics.csv'}")
    print(f"结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
