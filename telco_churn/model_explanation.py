"""模型解释模块：SHAP 和逻辑回归系数分析。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone


def _readable_feature_name(value):
    prefix, separator, bare = str(value).partition("__")
    return bare if separator and prefix in {"numeric", "categorical"} else str(value)


def explain_logistic_regression(model, X_train, y_train, feature_names, n_bootstrap=1000, random_state=42):
    """
    提取逻辑回归系数、OR 值及其 95% Bootstrap 置信区间。

    返回 DataFrame，包含 feature、coefficient、odds_ratio、ci_lower、ci_upper。
    """
    # 固定已拟合预处理的坐标系：类别列、参考组与标准差在重采样中不变。
    # 这是条件于预处理的惩罚 LR 描述区间，不是因果 OR 或无惩罚 MLE 区间。
    lr_model = model.named_steps["model"]
    coef = lr_model.coef_[0].copy()
    transformed = model.named_steps["preprocess"].transform(X_train)
    labels = np.asarray(y_train)
    if n_bootstrap < 2 or len(labels) != transformed.shape[0] or np.unique(labels).size != 2:
        raise ValueError("系数 Bootstrap 需要两类等长标签及至少两次重采样。")
    if len(feature_names) != len(coef):
        raise ValueError("特征名称与已拟合系数维度不一致。")
    rng = np.random.default_rng(random_state)
    bootstrap_coefs = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[indices]).size < 2:
            continue
        model_copy = clone(lr_model)
        transformed_sample = transformed.iloc[indices] if hasattr(transformed, "iloc") else transformed[indices]
        model_copy.fit(transformed_sample, labels[indices])
        bootstrap_coefs.append(model_copy.coef_[0])
    if len(bootstrap_coefs) < 2:
        raise ValueError("有效系数 Bootstrap 次数不足。")
    bootstrap_coefs = np.asarray(bootstrap_coefs)
    ci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)

    vif_values = _variance_inflation_factors(transformed)
    vif_notes = ["高共线，OR 解释需谨慎" if np.isfinite(value) and value > 10
                 or not np.isfinite(value) else "" for value in vif_values]

    # 构建结果表
    results = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "odds_ratio": np.exp(coef),
        "ci_lower": np.exp(ci_lower),
        "ci_upper": np.exp(ci_upper),
        "VIF": vif_values,
        "共线性提示": vif_notes,
        "有效Bootstrap次数": len(bootstrap_coefs),
        "区间口径": "固定预处理的惩罚LR描述区间，非因果",
        "系数口径": ["数值特征每训练集1SD" if str(name).startswith("numeric__")
                     else "相对于独热编码参考组" for name in feature_names],
    })

    # 按系数绝对值排序
    results = results.reindex(results["coefficient"].abs().sort_values(ascending=False).index)

    return results


def _variance_inflation_factors(matrix):
    """按设计矩阵计算 VIF；无法估计的精确共线列返回无穷。"""
    values = np.asarray(matrix, dtype=float)
    factors = []
    for index in range(values.shape[1]):
        target = values[:, index]
        others = np.delete(values, index, axis=1)
        if np.nanstd(target) == 0:
            factors.append(np.inf)
            continue
        design = np.column_stack([np.ones(len(values)), others])
        fitted, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ fitted
        total = target - target.mean()
        denominator = float(total @ total)
        r_squared = 1.0 - float(residual @ residual) / denominator if denominator else 1.0
        if r_squared >= 1 - 1e-10:
            factors.append(np.inf)
        else:
            factors.append(float(1.0 / max(1e-12, 1.0 - r_squared)))
    return factors


def explain_shap(model, X_test, feature_names, output_dir, max_display=20, X_test_raw=None):
    """
    使用 SHAP TreeExplainer 解释 LightGBM 或随机森林模型。

    保存 summary plot 和 top 样本的 waterfall plot。
    """
    import shap

    # 获取模型
    tree_model = model.named_steps["model"]

    # 转换测试集
    X_test_transformed = model.named_steps["preprocess"].transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    if X_test_raw is None:
        X_test_raw = X_test
    preprocess = model.named_steps["preprocess"]
    numeric_columns = list(preprocess.transformers_[0][2])
    display_data = pd.DataFrame(X_test_transformed, columns=feature_names)
    for index, column in enumerate(numeric_columns):
        display_data.iloc[:, index] = X_test_raw[column].to_numpy()
    if display_data.shape[1] > len(numeric_columns):
        display_data.iloc[:, len(numeric_columns):] = np.rint(
            display_data.iloc[:, len(numeric_columns):]).astype(int)
    display_feature_names = [_readable_feature_name(name) for name in feature_names]
    display_data.columns = display_feature_names

    # 创建 SHAP explainer
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test_transformed)

    # 如果返回的是列表（二分类），取正类的 SHAP 值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=display_feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Waterfall plots for the 20 highest-scoring customers
    (output_dir / "shap_examples").mkdir(exist_ok=True)

    # 按预测概率排序，取前 20
    probabilities = tree_model.predict_proba(X_test_transformed)[:, 1]
    top_indices = np.argsort(-probabilities)[:20]

    expected_value = np.asarray(explainer.expected_value)
    base_value = float(expected_value[-1]) if expected_value.ndim else float(expected_value)

    for i, idx in enumerate(top_indices):
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=base_value,
                data=display_data.iloc[idx].to_numpy(),
                feature_names=display_feature_names
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_examples" / f"waterfall_top{i+1}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    return shap_values
