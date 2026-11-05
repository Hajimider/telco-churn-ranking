"""模型解释模块：SHAP 和逻辑回归系数分析。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def explain_logistic_regression(model, X_train, y_train, feature_names, n_bootstrap=1000, random_state=42):
    """
    提取逻辑回归系数、OR 值及其 95% Bootstrap 置信区间。

    返回 DataFrame，包含 feature、coefficient、odds_ratio、ci_lower、ci_upper。
    """
    # 获取拟合好的逻辑回归模型
    lr_model = model.named_steps["model"]
    coef = lr_model.coef_[0]

    # Bootstrap 估计置信区间
    np.random.seed(random_state)
    bootstrap_coefs = []

    n_samples = len(X_train)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train.iloc[indices]
        y_boot = y_train.iloc[indices]

        try:
            model_copy = model.__class__(model.steps)
            model_copy.fit(X_boot, y_boot)
            bootstrap_coefs.append(model_copy.named_steps["model"].coef_[0])
        except:
            continue

    bootstrap_coefs = np.array(bootstrap_coefs)
    ci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)

    # 构建结果表
    results = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "odds_ratio": np.exp(coef),
        "ci_lower": np.exp(ci_lower),
        "ci_upper": np.exp(ci_upper),
    })

    # 按系数绝对值排序
    results = results.reindex(results["coefficient"].abs().sort_values(ascending=False).index)

    return results


def explain_shap(model, X_test, feature_names, output_dir, max_display=20):
    """
    使用 SHAP TreeExplainer 解释 LightGBM 或随机森林模型。

    保存 summary plot 和 top 样本的 waterfall plot。
    """
    import shap

    # 获取模型
    tree_model = model.named_steps["model"]

    # 转换测试集
    X_test_transformed = model.named_steps["preprocess"].transform(X_test)

    # 创建 SHAP explainer
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test_transformed)

    # 如果返回的是列表（二分类），取正类的 SHAP 值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Waterfall plots for top 5 high-risk customers
    (output_dir / "shap_examples").mkdir(exist_ok=True)

    # 按预测概率排序，取前 5
    probabilities = tree_model.predict_proba(X_test_transformed)[:, 1]
    top_indices = np.argsort(-probabilities)[:5]

    for i, idx in enumerate(top_indices):
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1],
                data=X_test_transformed[idx],
                feature_names=feature_names
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_examples" / f"waterfall_top{i+1}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    return shap_values
