"""LightGBM 超参数优化模块。"""
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def tune_lightgbm(X_train, y_train, n_trials=100, random_state=42, n_jobs=-1):
    """使用 Optuna 优化 LightGBM 超参数，目标是最大化 5 折 CV 的 AP 均值。"""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": random_state,
            "n_jobs": 1,  # 交叉验证已经并行，模型内部单线程
            "verbosity": -1,
        }

        # 处理类别不平衡
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params["scale_pos_weight"] = pos_weight

        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, scoring="average_precision",
                                  cv=cv, n_jobs=n_jobs)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    best_params = study.best_params.copy()
    best_params["random_state"] = random_state
    best_params["n_jobs"] = -1
    best_params["verbosity"] = -1

    # 加上 scale_pos_weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    best_params["scale_pos_weight"] = pos_weight

    return best_params, study.best_value, study
