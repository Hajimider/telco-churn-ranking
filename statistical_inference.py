"""Bootstrap 置信区间和模型比较统计检验。"""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def bootstrap_metrics(y_true, y_pred_proba, n_bootstrap=1000, random_state=42):
    """
    对测试集指标进行 Bootstrap 估计 95% 置信区间。

    返回字典，包含 AP、Precision、Recall、F1 的均值和置信区间。
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    ap_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]
        y_pred_class = (y_pred_boot >= 0.5).astype(int)

        ap_scores.append(average_precision_score(y_boot, y_pred_boot))
        precision_scores.append(precision_score(y_boot, y_pred_class, zero_division=0))
        recall_scores.append(recall_score(y_boot, y_pred_class, zero_division=0))
        f1_scores.append(f1_score(y_boot, y_pred_class, zero_division=0))

    def get_ci(scores):
        return {
            "mean": float(np.mean(scores)),
            "ci_lower": float(np.percentile(scores, 2.5)),
            "ci_upper": float(np.percentile(scores, 97.5))
        }

    return {
        "AP": get_ci(ap_scores),
        "Precision": get_ci(precision_scores),
        "Recall": get_ci(recall_scores),
        "F1": get_ci(f1_scores),
    }


def compare_models_paired_t_test(cv_scores_1, cv_scores_2, model_name_1, model_name_2):
    """
    使用配对 t 检验比较两个模型的交叉验证 AP 差异。

    cv_scores_1, cv_scores_2: 5 折 CV 的 AP 分数数组
    返回：t 统计量、p 值、均值差异
    """
    scores_1 = np.array(cv_scores_1)
    scores_2 = np.array(cv_scores_2)

    t_stat, p_value = stats.ttest_rel(scores_1, scores_2)

    return {
        "模型1": model_name_1,
        "模型2": model_name_2,
        "模型1均值": float(np.mean(scores_1)),
        "模型2均值": float(np.mean(scores_2)),
        "均值差异": float(np.mean(scores_1) - np.mean(scores_2)),
        "t统计量": float(t_stat),
        "p值": float(p_value),
        "显著性": "显著" if p_value < 0.05 else "不显著"
    }
