"""LightGBM 超参数优化模块。"""
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def tune_lightgbm(X_train, y_train, n_trials=100, random_state=42, n_jobs=-1,
                  estimator_factory=None, n_splits=3):
    """只在调用方传入的训练子集内搜索；最佳内层分数不作外层评估。"""
    if n_trials < 1 or n_splits < 2:
        raise ValueError("调参次数至少为1，内层折数至少为2。")

    def objective(trial):
        depth = trial.suggest_int("max_depth", 3, 8)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": depth,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, min(64, 2 ** depth)),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": random_state,
            "n_jobs": 1,  # 交叉验证已经并行，模型内部单线程
            "verbosity": -1,
        }

        # 不使用类别重权重；概率校准只在当前训练折内部完成。
        model = estimator_factory(params) if estimator_factory else LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, scoring="average_precision",
                                  cv=cv, n_jobs=n_jobs)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

    best_params = study.best_params.copy()
    best_params["random_state"] = random_state
    best_params["n_jobs"] = 1
    best_params["subsample_freq"] = 1
    best_params["verbosity"] = -1

    return best_params, study.best_value, study
