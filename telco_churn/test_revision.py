"""核心修复的快速回归测试；不需要数据库或网络。"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from .budget_analysis import budget_table, select_budget, business_scenarios
from .model_explanation import explain_logistic_regression
from .mysql_segments import analyze_training_segments
from .pipeline import make_pipeline, engineer_features, select_f1_threshold
from .statistical_inference import bootstrap_metrics, compare_models_paired_t_test


def test_tune_model_runs_real_logistic_and_random_forest_searches():
    from . import pipeline, validation
    X = pd.DataFrame({"value": np.linspace(-2, 2, 24)})
    for column in pipeline.CATEGORICAL_COLUMNS:
        X[column] = "No"
    y = pd.Series([0, 1] * 12)
    for name in ("逻辑回归", "随机森林"):
        model = validation.tune_model(name, X, y, ["value"], seed=7,
                                       trials=1, inner_splits=2)
        model.fit(X, y)
        assert np.isfinite(model.predict_proba(X)).all()


def test_styled_diagnostics_can_render_from_saved_tables(tmp_path):
    from .reporting import render_diagnostics
    calibration, pr, comparison = [], [], []
    for name in ["逻辑回归", "随机森林", "LightGBM"]:
        for state in ["原始", "sigmoid"]:
            calibration.extend({"模型": name, "状态": state, "平均预测": p, "实际比例": q}
                               for p, q in [(.1, .15), (.4, .35), (.8, .75)])
            comparison.append({"模型": name, "状态": state, "AP": .65, "Brier": .13})
        pr.extend({"模型": name, "Recall": r, "Precision": p}
                  for r, p in [(0, 1), (.5, .7), (1, .26)])
    render_diagnostics(pd.DataFrame(calibration), pd.DataFrame(pr),
                       pd.DataFrame(comparison), .26, tmp_path)
    assert (tmp_path / "calibration_pr.png").stat().st_size > 10000


def test_budget_allows_no_contact_and_uses_individual_value():
    table = budget_table([1, 0], [.9, .1], ["a", "b"], [1, 100],
                         k_range=[0, .5, 1], cost_per_contact=50)
    assert select_budget(table) == 0
    table = budget_table([1, 0], [.9, .1], ["a", "b"], [100, 1],
                         k_range=[0, .5, 1], cost_per_contact=50)
    assert select_budget(table) == .5
    assert table.iloc[1]["期望价值"] == pytest.approx(100 * 12 * .3)
    assert len(business_scenarios([1, 0], [.9, .1], ["a", "b"], [100, 1])) == 27


def test_budget_tie_break_and_input_validation():
    table = budget_table([1, 0], [.5, .5], ["b", "a"], [100, 1], k_range=[.5])
    assert table.loc[table["K比例"].eq(.5), "命中流失"].iloc[0] == 0
    with pytest.raises(ValueError):
        budget_table([0, 1], [np.inf, .5], ["a", "b"], [20, 20])
    with pytest.raises(ValueError):
        budget_table([0, 2], [.3, .5], ["a", "b"], [20, 20])


def test_corrected_comparison_formula():
    a, b = np.array([.5, .6, .9, .7]), np.array([.4, .65, .7, .65])
    result = compare_models_paired_t_test(a, b, "a", "b", test_train_ratio=.25)
    expected = np.sqrt((1 / len(a) + .25) * np.var(a - b, ddof=1))
    assert result["修正标准误"] == pytest.approx(expected)
    assert result["差值区间下限"] < result["均值差异"] < result["差值区间上限"]
    assert compare_models_paired_t_test(a, a, "a", "a")["p值"] == 1


def test_bootstrap_does_not_touch_global_random_state():
    np.random.seed(19)
    before = np.random.get_state()
    result = bootstrap_metrics([0, 1, 0, 1], [.1, .8, .4, .7], n_bootstrap=20)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert result["AP"]["有效次数"] >= 2


def test_bootstrap_reports_fixed_capacity_metrics_deterministically():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores = np.array([.1, .9, .3, .8, .2, .7, .4, .6])
    ids = [f"C{i}" for i in range(len(y))]
    first = bootstrap_metrics(y, scores, n_bootstrap=80, random_state=17,
                              budget_ratio=.25, sample_ids=ids)
    second = bootstrap_metrics(y, scores, n_bootstrap=80, random_state=17,
                               budget_ratio=.25, sample_ids=ids)
    assert first == second
    assert set(first) == {
        "AP", "ROC-AUC", "Brier", "Precision", "Recall", "F1",
        "Precision@K", "Recall@K",
    }
    assert all(0 <= interval["ci_lower"] <= interval["ci_upper"] <= 1
               for interval in first.values())
    with pytest.raises(ValueError, match="固定名额比例"):
        bootstrap_metrics(y, scores, n_bootstrap=10, budget_ratio=.01)


def test_bootstrap_topk_ties_follow_customer_id_order():
    y = np.array([1, 0, 1, 0])
    scores = np.full(4, .5)
    ids = np.array(["d", "a", "c", "b"])
    seed, iterations = 31, 100
    result = bootstrap_metrics(y, scores, n_bootstrap=iterations, random_state=seed,
                               budget_ratio=.25, sample_ids=ids)
    rng = np.random.default_rng(seed)
    expected = []
    for indices in rng.integers(0, len(y), size=(iterations, len(y))):
        labels = y[indices]
        if np.unique(labels).size < 2:
            continue
        selected = indices[np.lexsort((ids[indices], -scores[indices]))[0]]
        expected.append(y[selected])
    assert result["Precision@K"]["mean"] == pytest.approx(np.mean(expected))


def test_logistic_bootstrap_preserves_fitted_model_and_feature_coordinates():
    X = pd.DataFrame({"value": np.arange(40, dtype=float), "category": ["a"] * 39 + ["rare"]})
    y = pd.Series([0, 1] * 20)
    model = make_pipeline(LogisticRegression(max_iter=1000), ["value"], ["category"]).fit(X, y)
    names = model.named_steps["preprocess"].get_feature_names_out()
    before = model.predict_proba(X).copy()
    coefficients = model.named_steps["model"].coef_.copy()
    result = explain_logistic_regression(model, X, y, names, n_bootstrap=10)
    np.testing.assert_array_equal(model.predict_proba(X), before)
    np.testing.assert_array_equal(model.named_steps["model"].coef_, coefficients)
    assert len(result) == len(names)
    assert (result["有效Bootstrap次数"] == 10).all()


def test_sqlite_segments_need_no_credentials(monkeypatch):
    monkeypatch.delenv("CHURN_MYSQL_PASSWORD", raising=False)
    frame = pd.DataFrame({"Contract": ["Month-to-month", "Two year", "Two year"],
                          "tenure": [6, 48, 60], "MonthlyCharges": [90, 60, 30], "Churn": [1, 0, 0]})
    overall, groups, cross = analyze_training_segments(frame)
    assert overall.iloc[0]["客户数"] == 3
    assert overall.iloc[0]["流失率"] == pytest.approx(1 / 3)
    assert groups["客户数"].sum() == 9
    assert cross["客户数"].sum() == 3


def test_sqlite_pareto_exports_monthly_exposure(monkeypatch):
    monkeypatch.delenv("CHURN_MYSQL_PASSWORD", raising=False)
    frame = pd.DataFrame({"Contract": ["Month-to-month", "Two year", "Two year"],
                          "tenure": [6, 48, 60], "MonthlyCharges": [90, 60, 30],
                          "Churn": [1, 0, 0]})
    *_, pareto = analyze_training_segments(frame, include_pareto=True)
    assert "流失月费敞口" in pareto.columns
    assert pareto.iloc[0]["流失月费敞口"] >= pareto.iloc[-1]["流失月费敞口"]


def test_threshold_rejects_fractional_labels_and_matches_brute_force():
    from sklearn.metrics import f1_score
    with pytest.raises(ValueError):
        select_f1_threshold([0, 1.5, 1], [.1, .8, .6])
    labels = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([.1, .4, .4, .6, .8, .9])
    candidates = np.unique(np.r_[0, .5, 1, scores])
    values = np.array([f1_score(labels, scores >= t) for t in candidates])
    expected = min(candidates[np.isclose(values, values.max())],
                   key=lambda t: (abs(t - .5), -t))
    assert select_f1_threshold(labels, scores) == expected


def test_nested_validation_keeps_outer_validation_out_of_fitting(monkeypatch):
    from . import validation
    X = pd.DataFrame({"row_id": np.arange(40), "value": np.arange(40) % 7})
    y = pd.Series([0, 1] * 20)
    calls = []

    class CheckedModel:
        def __init__(self, training_ids):
            self.training_ids = set(training_ids)

        def predict_proba(self, frame):
            assert self.training_ids.isdisjoint(set(frame.row_id))
            scores = .1 + .8 * (frame.value.to_numpy() / 6)
            return np.column_stack([1 - scores, scores])

    def checked_pair(name, training, labels, columns, seed, trials, inner_splits):
        assert training.index.equals(labels.index)
        calls.append((name, tuple(training.row_id), tuple(columns)))
        model = CheckedModel(training.row_id)
        return model, model

    monkeypatch.setattr(validation, "fit_pair", checked_pair)
    groups = {"原始特征": ["value"], "原始+费用": ["value"],
              "原始+服务": ["value"], "全部衍生": ["value"]}
    result = validation.nested_validation(X, y, groups, outer_splits=2,
                                          repeats=2, inner_splits=2, trials=1)
    assert len(calls) == 4 * 6
    assert len(result["folds"]) == 4 * 3
    assert len(result["ablation"]) == 4 * 4
    assert all(len(values) == 40 and np.isfinite(values).all()
               for values in result["oof"].values())
    assert all(row["校正后p值"] >= row["p值"] for row in result["comparisons"])


def test_fit_pair_returns_the_calibrated_full_training_estimator(monkeypatch):
    from . import validation
    X = pd.DataFrame({"value": np.arange(40, dtype=float)})
    y = pd.Series([0, 1] * 20)
    tuned_rows = []

    def tune(name, subset, labels, columns, seed, trials, inner_splits):
        tuned_rows.append(tuple(subset.index))
        return make_pipeline(LogisticRegression(max_iter=1000), columns, [])

    monkeypatch.setattr(validation, "tune_model", tune)
    raw, calibrated = validation.fit_pair("逻辑回归", X, y, ["value"], 42, 1, 2)
    assert tuned_rows == [tuple(X.index)]
    assert raw.predict_proba(X).shape == (40, 2)
    assert np.isfinite(calibrated.predict_proba(X)).all()


def test_pipeline_freezes_decisions_before_test_labels(monkeypatch, tmp_path):
    from . import pipeline, validation
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    count = 100
    frame = pd.DataFrame({name: ["Yes"] * count for name in pipeline.CATEGORICAL_COLUMNS})
    frame["customerID"] = [f"C{i:04d}" for i in range(count)]
    frame["SeniorCitizen"] = np.arange(count) % 2
    frame["tenure"] = np.arange(count) % 30 + 1
    frame["MonthlyCharges"] = 30 + np.arange(count) % 50
    frame["TotalCharges"] = frame.tenure * frame.MonthlyCharges
    frame["Churn"] = np.where(np.arange(count) % 2, "Yes", "No")
    frame["Contract"] = "Month-to-month"
    frame["InternetService"] = "DSL"
    frame["PaymentMethod"] = "Electronic check"
    split = (np.arange(80), np.arange(80, count))
    monkeypatch.setattr(pipeline, "train_test_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)

    def small_model(name, X, y, numeric_columns, seed, trials, inner_splits):
        models = {
            "逻辑回归": LogisticRegression(max_iter=1000, random_state=seed),
            "随机森林": RandomForestClassifier(n_estimators=5, random_state=seed),
            "LightGBM": LGBMClassifier(n_estimators=5, num_leaves=4, min_child_samples=2,
                                        random_state=seed, n_jobs=1, verbosity=-1),
        }
        return make_pipeline(models[name], numeric_columns)

    monkeypatch.setattr(validation, "tune_model", small_model)
    options = dict(outer_splits=2, repeats=1, inner_splits=2, trials=1,
                   n_bootstrap=10, explanations=False)
    original = pipeline.run_pipeline(frame.copy(), tmp_path / "first", **options)
    frame.loc[80:, "Churn"] = frame.loc[80:, "Churn"].map({"Yes": "No", "No": "Yes"})
    changed = pipeline.run_pipeline(frame, tmp_path / "second", **options)
    assert original["交叉验证"] == changed["交叉验证"]
    assert original["阈值评估"] == changed["阈值评估"]
    assert original["名额敏感性"]["训练OOF选择名额比例"] == changed["名额敏感性"]["训练OOF选择名额比例"]
    assert (tmp_path / "first" / "metrics.json").is_file()
    assert original["报告版本"] == 4
    intervals = original["模型"]["LightGBM"]["Bootstrap置信区间"]
    assert {"ROC-AUC", "Brier", "Precision@K", "Recall@K"}.issubset(intervals)
    assert len(pd.read_csv(tmp_path / "first" / "business_scenarios.csv")) == 27
    assert original["生存分析"]["状态"] == "默认禁用"
    assert any(not decision["使用校准模型"]
               for decision in original["校准决策"].values())


def test_nested_validation_rejects_insufficient_minority():
    from .validation import nested_validation
    X = pd.DataFrame({"value": np.arange(20)})
    y = pd.Series([0] * 18 + [1] * 2)
    with pytest.raises(ValueError, match="外层折数"):
        nested_validation(X, y, {"全部衍生": ["value"]}, outer_splits=3,
                          repeats=1, inner_splits=2, trials=1)
