"""固定模型的测试集 Bootstrap 与相关交叉验证得分的近似比较。"""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def bootstrap_metrics(y_true, y_pred_proba, n_bootstrap=1000, random_state=42, threshold=0.5):
    """条件于已训练模型的百分位区间，不包含训练/选模不确定性。"""
    y = np.asarray(y_true)
    p = np.asarray(y_pred_proba, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("标签与分数必须是等长非空一维数组。")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Bootstrap 需要两类二值标签。")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("概率必须有限且位于 [0, 1]。")
    if (not isinstance(n_bootstrap, (int, np.integer)) or isinstance(n_bootstrap, bool)
            or n_bootstrap < 2 or not 0 <= threshold <= 1):
        raise ValueError("Bootstrap 次数至少为 2，阈值须在 [0, 1]。")
    rng = np.random.default_rng(random_state)
    values = {name: [] for name in ("AP", "Precision", "Recall", "F1")}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), len(y))
        labels, scores = y[indices], p[indices]
        if np.unique(labels).size < 2:
            continue
        predicted = scores >= threshold
        values["AP"].append(average_precision_score(labels, scores))
        for name, function in [("Precision", precision_score), ("Recall", recall_score), ("F1", f1_score)]:
            values[name].append(function(labels, predicted, zero_division=0))
    if len(values["AP"]) < 2:
        raise ValueError("有效 Bootstrap 次数不足，请增加重采样次数。")
    return {name: {"mean": float(np.mean(scores)),
                   "ci_lower": float(np.percentile(scores, 2.5)),
                   "ci_upper": float(np.percentile(scores, 97.5)),
                   "有效次数": len(scores)} for name, scores in values.items()}


def compare_models_paired_t_test(cv_scores_1, cv_scores_2, model_name_1, model_name_2,
                                test_train_ratio=0.25):
    """Nadeau–Bengio corrected resampled t 近似；不是独立折 t 检验。

    N 次配对分数的差异方差估计为 (1/N + n_valid/n_train)*s²。
    区间是单组比较的近似区间，多重比较的 p 值由调用方 Holm 校正。
    """
    a, b = np.asarray(cv_scores_1, float), np.asarray(cv_scores_2, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("需要至少两个等长的配对得分。")
    if (not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(test_train_ratio) or test_train_ratio <= 0):
        raise ValueError("得分必须有限，验证/训练样本比必须为正。")
    difference = a - b
    mean = float(difference.mean())
    se = float(np.sqrt((1 / len(a) + test_train_ratio) * difference.var(ddof=1)))
    if se == 0:
        # 常数非零差异无法从数据估计方差，不制造无限 t 与 p=0。
        t_value, p_value = 0.0, 1.0
    else:
        t_value = mean / se
        p_value = float(2 * stats.t.sf(abs(t_value), df=len(a) - 1))
    margin = float(stats.t.ppf(0.975, len(a) - 1) * se)
    return {"模型1": model_name_1, "模型2": model_name_2,
            "模型1均值": float(a.mean()), "模型2均值": float(b.mean()),
            "均值差异": mean, "t统计量": float(t_value), "p值": p_value,
            "差值区间下限": mean - margin if se else None,
            "差值区间上限": mean + margin if se else None,
            "修正标准误": se, "方法": "Nadeau-Bengio 近似修正",
            "检验状态": "差异方差为零，不作显著判断" if se == 0 else "近似检验，待 Holm 校正"}
