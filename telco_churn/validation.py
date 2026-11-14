"""嵌套重复验证：外层验证标签不参与调参、校准或预处理拟合。"""
import logging
from itertools import combinations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from .lightgbm_tuning import tune_lightgbm
from .statistical_inference import compare_models_paired_t_test

LOGGER = logging.getLogger(__name__)
MODEL_NAMES = ["逻辑回归", "随机森林", "LightGBM"]


def tune_model(name, X, y, numeric_columns, seed, trials, inner_splits):
    from .pipeline import make_pipeline
    cv = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    if name == "LightGBM":
        params, _, _ = tune_lightgbm(
            X, np.asarray(y), n_trials=trials, n_jobs=1, n_splits=inner_splits,
            random_state=seed,
            estimator_factory=lambda p: make_pipeline(LGBMClassifier(**p), numeric_columns))
        return make_pipeline(LGBMClassifier(**params), numeric_columns)
    if name == "逻辑回归":
        estimator = LogisticRegression(max_iter=2000, random_state=seed)
        grid = {"model__C": [0.1, 1.0, 10.0]}
    else:
        estimator = RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=seed)
        grid = {"model__min_samples_leaf": [2, 5, 10]}
    search = GridSearchCV(make_pipeline(estimator, numeric_columns), grid,
                          scoring="average_precision", cv=cv, n_jobs=1, refit=False,
                          error_score="raise")
    search.fit(X, y)
    return make_pipeline(estimator, numeric_columns).set_params(**search.best_params_)


def fit_pair(name, X, y, numeric_columns, seed, trials, inner_splits):
    """外层训练集内调参，再交叉拟合 sigmoid；原始模型仅作校准前对照。

    校准的 OOF 仅对基模型拟合隔离，超参数在当前训练子集选择，未做第三层
    搜索隔离。校准效果必须由外层验证评估，不以内部拟合分数证明改善。
    """
    estimator = tune_model(name, X, y, numeric_columns, seed, trials, inner_splits)
    cv = StratifiedKFold(inner_splits, shuffle=True, random_state=seed + 1)
    calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv, ensemble=False)
    calibrated.fit(X, y)
    # ensemble=False 已在全部当前训练数据上重拟合基模型，无需再拟合一次。
    raw = calibrated.calibrated_classifiers_[0].estimator
    return raw, calibrated


def nested_validation(X, y, feature_groups, *, outer_splits=5, repeats=3,
                      inner_splits=3, trials=30, seed=42):
    """消融固定使用 LR 家族，所有特征组与家族共用相同外层划分。

    调参/校准全部位于外层训练数据内；外层分数用于比较候选流程而非盲测。
    重复 CV 得分相关，因此标准差是折间波动，不是标准误或置信区间。
    """
    settings = (outer_splits, inner_splits, repeats, trials)
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in settings):
        raise ValueError("折数、重复次数及调参次数必须为整数。")
    if outer_splits < 2 or inner_splits < 2 or repeats < 1 or trials < 1:
        raise ValueError("外/内层折数至少为2，重复及调参次数至少为1。")
    labels = np.asarray(y)
    if labels.ndim != 1 or len(X) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("嵌套验证需要等长的一维二值标签。")
    if np.unique(labels).size != 2 or np.bincount(labels.astype(int)).min() < outer_splits:
        raise ValueError("训练集每类样本数均须不少于外层折数。")
    outer = list(RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=repeats,
                                        random_state=seed).split(X, y))
    if any(pd.Series(np.asarray(y)[train]).value_counts().min() < inner_splits for train, _ in outer):
        raise ValueError("外层训练折的少数类不足以支持内层分层验证。")
    oof = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    raw_oof = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    counts = np.zeros(len(y), dtype=int)
    scores = {name: [] for name in MODEL_NAMES}
    auc_scores = {name: [] for name in MODEL_NAMES}
    fold_rows, ablation_rows = [], []
    columns = feature_groups["全部衍生"]
    ratios = []
    for fold, (train, valid) in enumerate(outer):
        LOGGER.info("外层折 %s/%s", fold + 1, len(outer))
        train_X, valid_X = X.iloc[train], X.iloc[valid]
        train_y, valid_y = y.iloc[train], y.iloc[valid]
        counts[valid] += 1
        ratios.append(len(valid) / len(train))
        for name in MODEL_NAMES:
            raw, calibrated = fit_pair(name, train_X, train_y, columns, seed + fold,
                                       trials, inner_splits)
            p = calibrated.predict_proba(valid_X)[:, 1]
            p_raw = raw.predict_proba(valid_X)[:, 1]
            oof[name][valid] += p
            raw_oof[name][valid] += p_raw
            ap = float(average_precision_score(valid_y, p))
            auc = float(roc_auc_score(valid_y, p))
            scores[name].append(ap)
            auc_scores[name].append(auc)
            fold_rows.append({"外层折": fold, "重复": fold // outer_splits,
                              "模型": name, "AP": ap, "ROC-AUC": auc,
                              "原始AP": float(average_precision_score(valid_y, p_raw)),
                              "校准Brier": float(brier_score_loss(valid_y, p)),
                              "原始Brier": float(brier_score_loss(valid_y, p_raw))})
            if name == "逻辑回归":
                ablation_rows.append({"外层折": fold, "特征组": "全部衍生", "AP": ap})
        for group, subset in feature_groups.items():
            if group == "全部衍生":
                continue
            _, calibrated = fit_pair("逻辑回归", train_X, train_y, subset,
                                      seed + fold, trials, inner_splits)
            ablation_rows.append({"外层折": fold, "特征组": group,
                                  "AP": float(average_precision_score(valid_y,
                                                  calibrated.predict_proba(valid_X)[:, 1]))})
    if not np.all(counts == repeats):
        raise RuntimeError("外层折外预测覆盖次数异常。")
    comparisons = [compare_models_paired_t_test(scores[a], scores[b], a, b,
                                               test_train_ratio=float(np.mean(ratios)))
                   for a, b in combinations(MODEL_NAMES, 2)]
    from .pipeline import holm_adjust
    for row, adjusted in zip(comparisons, holm_adjust([r["p值"] for r in comparisons])):
        row["校正后p值"] = float(adjusted)
        row["显著性"] = "近似显著" if adjusted < .05 else "未检出显著差异"
    return {"scores": scores, "auc_scores": auc_scores,
            "oof": {name: p / counts for name, p in oof.items()},
            "raw_oof": {name: p / counts for name, p in raw_oof.items()},
            "folds": pd.DataFrame(fold_rows), "ablation": pd.DataFrame(ablation_rows),
            "comparisons": comparisons}
